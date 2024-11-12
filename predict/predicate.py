import torch
from model.modelNet3D import smallFocusNet
from utils.data_loader3D import load_LBW_data

label_name = ["T2", "T3", "T4"]

def _test_model(model_path, train_file, batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Load the trained model
    model = smallFocusNet(classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Load the dataset
    test_loader = load_LBW_data(train_file, batch_size=batch_size)

    total_targets = []
    total_preds = []

    # Evaluate the model
    with torch.no_grad():
        for images, origin_images, slice_paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            print("/".join(slice_paths[0][0].split("\\")[:-1]), ",Prediction label:", label_name[predictions.cpu().numpy()[0]])
            total_preds.extend(predictions.cpu().numpy())


if __name__ == "__main__":
    classes = 3
    model_path = '../models/2024-08-13-2/best_val_model.pth'  # Update this path
    annotations_file = r"E:\LBW\test3\processe\data_list.txt"
    _test_model(model_path, annotations_file)
