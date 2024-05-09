import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
import numpy as np
import h5py

# Model definition
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

# Load data
def load_data(path):
    with h5py.File(path, 'r') as file:
        test_set_x = file['test_set_x'][:].reshape(-1, 250, 100).astype('float32') / 255.0
        test_set_y = file['test_set_y'][:].astype('int64').flatten()

    test_data = TensorDataset(torch.tensor(test_set_x).unsqueeze(1), torch.tensor(test_set_y))

    num_samples = 500  # Number of samples you want to evaluate
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    subset_data = Subset(test_data, indices)
    return test_data

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    test_data = load_data('/mnt/c/Users/ashto/Desktop/Class/ME592/ME592X/GroupData/Combustion/Aditya_data/combustion_img_13.mat')
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    # Load the model
    model = CNN().to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
