import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader3D import load_trian_data, load_trian_val_data
import os
from model.modelNet3D import smallFocusNet
from torch.utils.tensorboard import SummaryWriter


def train_model(train_file, val_file, epochs=200, batch_size=4, save_path='./model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_loader, val_loader = load_trian_val_data(train_file, batch_size, random_state=124)
    val_loader = load_trian_data(val_file, batch_size)

    model = smallFocusNet(num_classes=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize TensorBoard
    writer = SummaryWriter()

    best_val_accuracy = 0
    lowest_train_loss = float('inf')

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct / total
        # _________________________________________________________________________
        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.eval()
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        # _________________________________________________________________________
        # Test the model
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(val_loader)
        test_accuracy = correct / total

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Save the model every 10 epochs
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

        # Save the model if it has the best validation accuracy
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{save_path}/best_val_model.pth")

        # Save the model if it has the lowest training loss
        if train_loss < lowest_train_loss:
            lowest_train_loss = train_loss
            torch.save(model.state_dict(), f"{save_path}/best_train_model.pth")

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Acc: {train_accuracy * 100:.2f}%, Val Loss: {val_loss}, Val Acc: {val_accuracy * 100:.2f}%, Test Loss: {test_loss}, Test Acc: {test_accuracy * 100:.2f}%")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train_file = r"C:\Users\just\Desktop\Workspace\train_test2\train\8bit\only_tumour\train_list.txt"  # Update this path
    test_file = r"C:\Users\just\Desktop\Workspace\train_test2\test\8bit\only_tumour\val_list.txt"
    save_path = 'models/2024-08-28-4rand_124'  # Update this path for saving models
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_model(train_file, test_file, epochs=200, save_path=save_path)
