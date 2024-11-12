import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_trian_data

def calculate_overall_auc(y_true, y_probs):
    y_true_bin = label_binarize(y_true, classes=np.arange(classes))
    n_classes = y_true_bin.shape[1]
    auc_scores = []

    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc_scores.append(auc(fpr, tpr))

    macro_auc = np.mean(auc_scores) if len(auc_scores) > 0 else 0
    return macro_auc


def bootstrap_overall_auc(y_true, y_probs, n_bootstraps=1000, alpha=0.05):
    """Calculate the confidence interval of the overall AUC using bootstrap method"""
    rng = np.random.RandomState(42)
    bootstrapped_auc_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = calculate_overall_auc(y_true[indices], y_probs[indices])
        bootstrapped_auc_scores.append(score)

    sorted_scores = np.sort(bootstrapped_auc_scores)
    lower = np.percentile(sorted_scores, 100 * alpha / 2)
    upper = np.percentile(sorted_scores, 100 * (1 - alpha / 2))

    return lower, upper



def bootstrap_confidence_interval(data, n_bootstraps=1000, alpha=0.05):
    """Computes the bootstrap confidence interval for a given data array."""
    rng = np.random.RandomState(42)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, len(data), len(data))
        if len(np.unique(data[indices])) < 2:
            continue
        bootstrapped_scores.append(np.mean(data[indices]))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower = np.percentile(sorted_scores, 100 * alpha / 2)
    upper = np.percentile(sorted_scores, 100 * (1 - alpha / 2))

    return lower, upper


def bootstrap_auc(y_true, y_prob, n_bootstraps=1000, alpha=0.05):
    """Computes the bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(42)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = auc(roc_curve(y_true[indices], y_prob[indices])[0], roc_curve(y_true[indices], y_prob[indices])[1])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower = np.percentile(sorted_scores, 100 * alpha / 2)
    upper = np.percentile(sorted_scores, 100 * (1 - alpha / 2))

    return lower, upper


def _test_model(model_path, train_file, classes=3, batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Load the trained model
    model = smallFocusNet(classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Load the dataset
    test_loader = load_trian_data(train_file, batch_size=batch_size)

    total_targets = []
    total_preds = []
    total_probs = []

    # Evaluate the model
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            total_targets.extend(labels.cpu().numpy())
            total_preds.extend(predictions.cpu().numpy())
            total_probs.extend(probs.cpu().numpy())



    # ROC and AUC for each class
    total_targets = label_binarize(total_targets, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_ci = dict()

    # Compute ROC curve, ROC area, and AUC confidence interval for each class
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(total_targets[:, i], np.array(total_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        auc_ci[i] = bootstrap_auc(total_targets[:, i], np.array(total_probs)[:, i])
        print(f'Class {i} AUC: {roc_auc[i]:.3f}, 95% CI: [{auc_ci[i][0]:.3f} - {auc_ci[i][1]:.3f}]')
    # Overall AUC
    overall_auc = calculate_overall_auc(np.array(total_targets), np.array(total_probs))
    overall_auc_ci = bootstrap_overall_auc(np.array(total_targets), np.array(total_probs))
    print(f'Overall AUC: {overall_auc:.3f}%, 95% CI: [{overall_auc_ci[0]:.3f} - {overall_auc_ci[1]:.3f}]')

if __name__ == "__main__":
    classes = 3
    model_path = 'E:\LBW\small_Focus_Net\models/2024-08-13-2/best_val_model.pth'  # Update this path
    annotations_file = r"E:\LBW\test3\processe\data_list.txt"  # Update this path
    _test_model(model_path, annotations_file)
