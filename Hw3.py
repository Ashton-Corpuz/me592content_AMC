import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=1, num_filters1=32, num_filters2=64, kernel_size1=3, kernel_size2=3,
                 stride1=1, stride2=1, padding1=1, padding2=1, pool_kernel_size=2, pool_stride=2,
                 num_units_fc1=128, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, num_filters1, kernel_size=kernel_size1,
                               stride=stride1, padding=padding1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=kernel_size2,
                               stride=stride2, padding=padding2)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        example_size = (1, num_channels, 250, 100)  # (batch_size, channels, height, width)
        with torch.no_grad():
            example_input = torch.autograd.Variable(torch.rand(example_size))
            example_output = self.pool(self.conv2(self.pool(self.conv1(example_input))))
            self.flattened_size = example_output.data.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.flattened_size, num_units_fc1)
        self.fc2 = nn.Linear(num_units_fc1, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(-1, self.flattened_size)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, valid_loader, num_epochs, device):
    best_accuracy = 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Saving new best model with accuracy {accuracy}%")
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, valid_data, _ = load_data('/mnt/c/Users/ashto/Desktop/Class/ME592/ME592X/GroupData/Combustion/Aditya_data/combustion_img_13.mat')
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=100, shuffle=False)

    model = SimpleCNN().to(device)
    num_epochs = 20
    train_model(model, train_loader, valid_loader, num_epochs, device)
