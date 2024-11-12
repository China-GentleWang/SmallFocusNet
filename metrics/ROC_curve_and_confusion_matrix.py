import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_trian_data
import matplotlib.pyplot as plt
import seaborn as sns  # 导入seaborn库


def _test_model(model_path, train_file, batch_size=1):
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

    # Calculate metrics
    cm = confusion_matrix(total_targets, total_preds)
    overall_accuracy = accuracy_score(total_targets, total_preds)

    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['T2', 'T3', 'T4a'], yticklabels=['T2', 'T3', 'T4a'])
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.show()

    # ROC and AUC for each class
    total_targets = label_binarize(total_targets, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(total_targets[:, i], np.array(total_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(total_targets.ravel(), np.array(total_probs).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))

    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    class_name = ["T2", "T3", "T4a"]
    for i in range(classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'ROC curve of class {class_name[i]} (area = {roc_auc[i]:.2f})')

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    classes = 3
    model_path = 'E:\LBW\small_Focus_Net\models/2024-08-13-2/best_val_model.pth'  # Update this path
    annotations_file = r"E:\LBW\test3\processe\data_list.txt"  # Update this path
    _test_model(model_path, annotations_file)
