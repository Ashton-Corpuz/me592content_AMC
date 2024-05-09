import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py

# Load data
def load_data(path):
    with h5py.File(path, 'r') as file:
        test_set_x = file['test_set_x'][:].reshape(-1, 250, 100).astype('float32') / 255.0
        test_set_y = file['test_set_y'][:].astype('int64').flatten()
        train_set_x = file['train_set_x'][:].reshape(-1, 250, 100).astype('float32') / 255.0
        train_set_y = file['train_set_y'][:].astype('int64').flatten()
        valid_set_x = file['valid_set_x'][:].reshape(-1, 250, 100).astype('float32') / 255.0
        valid_set_y = file['valid_set_y'][:].astype('int64').flatten()

    train_data = TensorDataset(torch.tensor(train_set_x).unsqueeze(1), torch.tensor(train_set_y))
    valid_data = TensorDataset(torch.tensor(valid_set_x).unsqueeze(1), torch.tensor(valid_set_y))
    test_data = TensorDataset(torch.tensor(test_set_x).unsqueeze(1), torch.tensor(test_set_y))

    return train_data, valid_data, test_data

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 62 * 25, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 62 * 25)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, valid_loader, num_epochs, device):
    best_accuracy = 0
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Adjusted learning rate for illustration
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Loss: {valid_loss / len(valid_loader)}, Accuracy: {accuracy}%')

        scheduler.step(valid_loss)  # Update the learning rate based on validation loss

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Saving new best model with accuracy {accuracy}%")
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, valid_data, _ = load_data('/mnt/c/Users/ashto/Desktop/Class/ME592/ME592X/GroupData/Combustion/Aditya_data/combustion_img_13.mat')
    train_loader = DataLoader(train_data, batch_size=40, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=40, shuffle=False)

    model = CNN().to(device)

    num_epochs = 100
    train_model(model, train_loader, valid_loader, num_epochs, device)
