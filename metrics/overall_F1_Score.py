import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_trian_data

def calculate_metrics(cm):
    """Calculate sensitivity, specificity, and F1 score for each class."""
    num_classes = cm.shape[0]
    f1_scores = np.zeros(num_classes)

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        f1_scores[i] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

    return np.mean(f1_scores)

def bootstrap_confidence_interval(y_true, y_pred, num_bootstraps=1000, alpha=0.05):
    """Calculate bootstrap confidence intervals for sensitivity, specificity, and F1 score."""
    rng = np.random.RandomState(42)

    bootstrapped_f1_scores = []

    for _ in range(num_bootstraps):
        indices = rng.choice(len(y_pred), len(y_pred), replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        cm = confusion_matrix(y_true_sample, y_pred_sample, labels=np.arange(np.max(y_true) + 1))
        f1_score_avg = calculate_metrics(cm)
        bootstrapped_f1_scores.append(f1_score_avg)

    f1_score_ci = np.percentile(bootstrapped_f1_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return  f1_score_ci

def _test_model(model_path, train_file, classes=3, batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = smallFocusNet(classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    test_loader = load_trian_data(train_file, batch_size=batch_size)
    total_targets = []
    total_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total_targets.extend(labels.cpu().numpy())
            total_preds.extend(predictions.cpu().numpy())

    cm = confusion_matrix(total_targets, total_preds)
    overall_f1_score = calculate_metrics(cm)
    f1_score_ci = bootstrap_confidence_interval(np.array(total_targets), np.array(total_preds))


    print(f'Overall F1 Score(overall F1 score): {overall_f1_score:.4f}, 95% CI: [{f1_score_ci[0]:.4f}, {f1_score_ci[1]:.4f}]')

if __name__ == "__main__":
    classes = 3
    model_path = 'models/2024-08-13-2/best_val_model.pth'  # 根据需要更新路径
    annotations_file = r"E:\LBW\test3\processe\data_list.txt"  # 根据需要更新路径
    _test_model(model_path, annotations_file)
