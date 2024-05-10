import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX
import h5py
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Load data
def load_data(path):
    transform = transforms.Compose([
    transforms.Resize((480, 640)),  # Resize all images to the expected input size of the CNN
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                        std=[0.229, 0.224, 0.225])
    ])

    data = ImageFolder(path, transform=transform)
    total = len(data)
    validation_num = int(0.2 * total)
    train_num = total - validation_num

    train_data, valid_data = random_split(data,[train_num,validation_num])

    return train_data, valid_data

class HKCNN(nn.Module):
    def __init__(self):
        super(HKCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)   
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8)) 
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)                    
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)                             

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))  
        x = F.relu(self.pool(self.conv2(x)))  
        x = F.relu(self.pool(self.conv3(x)))  
        x = self.adaptive_pool(x) 
        x = x.view(-1, 64 * 8 * 8) 
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
class TMCNN(nn.Module):
    def __init__(self):
        super(TMCNN, self).__init__() 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv4(x)))
        x = self.adaptive_pool(x) 
        x = self.dropout(x)
        x = x.view(-1, 512 * 4 * 4) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, valid_loader, num_epochs, device):
    best_accuracy = 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')
        summary_writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        summary_writer.add_scalar('Accuracy/Train', train_accuracy, epoch)


        model.eval()
        valid_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        accuracy = 100 * correct_val / total_val
        print(f'Validation Loss: {valid_loss / len(valid_loader)}, Accuracy: {accuracy}%')
        summary_writer.add_scalar('Loss/Val', (valid_loss / len(valid_loader)), epoch)
        summary_writer.add_scalar('Accuracy/Train', accuracy, epoch)

        scheduler.step(valid_loss)  # Update the learning rate based on validation loss

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Saving new best model with accuracy {accuracy}%")
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs16')

    train_data, valid_data = load_data('/mnt/c/Users/ashto/Desktop/Class/ME592/HW4/state-farm-distracted-driver-detection/imgs/train')
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers= 1)
    valid_loader = DataLoader(valid_data, batch_size=100, shuffle=False, num_workers= 1)

    model = HKCNN().to(device)

    num_epochs = 10
    train_model(model, train_loader, valid_loader, num_epochs, device)
