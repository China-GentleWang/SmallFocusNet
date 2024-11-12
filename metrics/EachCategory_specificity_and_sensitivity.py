import torch
import numpy as np
from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_test_data
from sklearn.metrics import confusion_matrix


def bootstrap_for_metrics(y_true, y_pred, num_classes, num_bootstraps=1000):
    rng = np.random.RandomState(42)  # 确保可重复性
    sensitivity_boots = np.zeros((num_bootstraps, num_classes))
    specificity_boots = np.zeros((num_bootstraps, num_classes))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for cls in range(num_classes):
        # 敏感性相关样本
        sensitivity_indices = np.where(y_true == cls)[0]
        y_true_sensitivity = y_true[sensitivity_indices]
        y_pred_sensitivity = y_pred[sensitivity_indices]

        # 特异性相关样本
        specificity_indices = np.where(y_true != cls)[0]
        y_true_specificity = y_true[specificity_indices]
        y_pred_specificity = y_pred[specificity_indices]

        for i in range(num_bootstraps):
            # 敏感性bootstrap
            sampled_indices = rng.choice(len(y_pred_sensitivity), len(y_pred_sensitivity), replace=True)
            y_true_sampled = y_true_sensitivity[sampled_indices]
            y_pred_sampled = y_pred_sensitivity[sampled_indices]
            y_true_binary = (y_true_sampled == cls).astype(int)
            y_pred_binary = (y_pred_sampled == cls).astype(int)
            cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
            sensitivity, _ = calculate_binary_metrics(cm)
            sensitivity_boots[i, cls] = sensitivity

            # 特异性bootstrap
            sampled_indices = rng.choice(len(y_pred_specificity), len(y_pred_specificity), replace=True)
            y_true_sampled = y_true_specificity[sampled_indices]
            y_pred_sampled = y_pred_specificity[sampled_indices]
            y_true_binary = (y_true_sampled != cls).astype(int)
            y_pred_binary = (y_pred_sampled != cls).astype(int)
            cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
            _, specificity = calculate_binary_metrics(cm)
            specificity_boots[i, cls] = specificity

    sensitivity_ci = np.percentile(sensitivity_boots, [2.5, 97.5], axis=0)
    specificity_ci = np.percentile(specificity_boots, [2.5, 97.5], axis=0)

    return sensitivity_ci, specificity_ci

def calculate_binary_metrics(cm):
    # Assuming binary confusion matrix for class `cls` vs. rest
    TP = cm[0, 0]
    FN = cm[0, 1]  # False Negatives: All non-TP in the same row
    FP = cm[1, 0]  # False Positives: All non-TN in the same column
    TN = cm[1, 1]  # True Negatives: Remaining part of the matrix
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0  # Recall
    specificity = TN / (TN + FP) if TN + FP != 0 else 0

    return sensitivity, specificity

def calculate_metrics(cm):
    num_classes = cm.shape[0]
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        sensitivity[i] = TP / (TP + FN) if TP + FN != 0 else 0
        specificity[i] = TN / (TN + FP) if TN + FP != 0 else 0
    return sensitivity, specificity
def display_metrics(sensitivity, specificity, sensitivity_ci, specificity_ci):
    labels = ['T2', 'T3', 'T4']
    for i in range(len(sensitivity)):
        print(f"类内指标：Class {labels[i]}:")
        print(f"  Sensitivity: {sensitivity[i]:.4f}, Sensitivity 95% CI: [{sensitivity_ci[0, i]:.4f}, {sensitivity_ci[1, i]:.4f}]")
        print(f"  Specificity: {specificity[i]:.4f}, Specificity 95% CI: [{specificity_ci[0, i]:.4f}, {specificity_ci[1, i]:.4f}]")


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
    print(cm)
    sensitivity, specificity = calculate_metrics(cm)
    sensitivity_ci, specificity_ci = bootstrap_for_metrics(total_targets, total_preds, classes)  # 获取F1置信区间

    display_metrics(sensitivity, specificity, sensitivity_ci, specificity_ci)  # 展示所有指标和置信区间
if __name__ == "__main__":
    classes = 3
    model_path = 'models/2024-08-13-2/best_val_model.pth'  # Update this path as needed
    annotations_file = r"E:\LBW\test3\processe\data_list.txt"  # Update this path as needed
    _test_model(model_path, annotations_file)
