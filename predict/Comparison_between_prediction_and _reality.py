import torch

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_trian_data


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

    # Evaluate the model
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # print(outputs)
            _, predictions = torch.max(outputs, 1)
            print("Prediction label:", predictions.cpu().numpy()[0], "real label:", labels.cpu().numpy()[0])
            total_targets.extend(labels.cpu().numpy())
            total_preds.extend(predictions.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(total_targets, total_preds)
    overall_accuracy = accuracy_score(total_targets, total_preds)

    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # Details for each class
    for i in range(len(cm)):
        class_total = np.sum(cm[i, :])
        true_positives = cm[i, i]
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        false_negatives = np.sum(cm[i, :]) - cm[i, i]
        true_negatives = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        class_accuracy = true_positives / class_total if class_total != 0 else 0

        print(f"Class {i} details:")
        print(f"  Total: {class_total}")
        print(f"  Correctly Predicted: {true_positives}")
        print(f"  Incorrectly Predicted: {class_total - true_positives}")
        print(f"  Class Accuracy: {class_accuracy * 100:.2f}%")
        print(f"  TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}")


if __name__ == "__main__":
    classes = 3
    model_path = 'models/2024-08-13-2/best_val_model.pth'  # Update this path
    annotations_file = r"E:\LBW\small_Focus_Net\dataset\test_list.txt"
    _test_model(model_path, annotations_file)
