from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_test_data
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
def display_metrics(f1_scores, f1_ci):
    labels = ['T2', 'T3', 'T4']
    for i in range(len(f1_scores)):
        print(f" Class {labels[i]}:")
        print(f"  F1-score: {f1_scores[i]:.4f}, F1-score 95% CI: [{f1_ci[i, 0]:.4f}, {f1_ci[i, 1]:.4f}]\n")


def calculate_metrics(y_true, y_pred, num_classes):
    f1_score = np.zeros(num_classes)

    for cls in range(num_classes):

        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]

        sensitivity = TP / (TP + FN) if TP + FN != 0 else 0  # Recall
        precision = TP / (TP + FP) if TP + FP != 0 else 0

        f1_score[cls] = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    return f1_score



def bootstrap_for_metrics(y_true, y_pred, num_classes, num_bootstraps=1000):
    rng = np.random.RandomState(42)
    f1_boots = np.zeros((num_bootstraps, num_classes))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for cls in range(num_classes):
        for i in range(num_bootstraps):
            # Bootstrap
            sampled_indices = rng.choice(len(y_true), len(y_true), replace=True)
            y_true_sampled = y_true[sampled_indices]
            y_pred_sampled = y_pred[sampled_indices]


            y_true_binary = (y_true_sampled == cls).astype(int)
            y_pred_binary = (y_pred_sampled == cls).astype(int)
            cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])

            # Calculate the binary F1 score of bootstrap samples
            TP = cm[0, 0]
            FN = cm[0, 1]
            FP = cm[1, 0]
            sensitivity = TP / (TP + FN) if TP + FN != 0 else 0  # Recall
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
            f1_boots[i, cls] = f1

    # Calculate the F1 score confidence interval for each category
    f1_ci = np.zeros((num_classes, 2))
    for cls in range(num_classes):
        lower_bound = np.percentile(f1_boots[:, cls], 2.5)
        upper_bound = np.percentile(f1_boots[:, cls], 97.5)
        f1_ci[cls] = [lower_bound, upper_bound]

    return f1_ci


def _test_model(model_path, train_file, classes=3, batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = smallFocusNet(classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    test_loader = load_test_data(train_file, batch_size=batch_size)
    total_targets = []
    total_preds = []

    with torch.no_grad():
        for images, labels, origin_images, slice_paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total_targets.extend(labels.cpu().numpy())
            total_preds.extend(predictions.cpu().numpy())

    cm = confusion_matrix(total_targets, total_preds)  # 混淆矩阵
    print("Confusion Matrix:\n", cm)

    # Calculate F1 score for each category using the One vs Rest method
    f1_score = calculate_metrics(np.array(total_targets), np.array(total_preds), classes)
    # print("F1-scores per class:", f1_score)

    # Calculate the F1 score confidence interval for each category using Bootstrap method
    f1_ci = bootstrap_for_metrics(np.array(total_targets), np.array(total_preds), classes)
    # print("F1-score Confidence Intervals per class:", f1_ci)

    # Display indicators and confidence intervals for each category
    display_metrics(f1_score, f1_ci)



if __name__ == "__main__":
    classes = 3
    model_path = 'models/2024-08-13-2/best_val_model.pth'  # Update this path as needed
    annotations_file = r"E:\LBW\test3\processe\data_list.txt"  # Update this path as needed
    _test_model(model_path, annotations_file)
