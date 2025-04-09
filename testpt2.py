import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# this transform will convert data to tensor then standardize the data (precomputed mean and std)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
# download training and testing and apply transform to both
training_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testing_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# these data loaders handle the data during training and testing. Increase testing batch size if you have a lot of RAM
trainloader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testing_data, batch_size=1000, shuffle=False)

print(testloader)
# for images, labels in testloader:
#     print(labels)
# assert 1 == 0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class MyCNN(nn.Module):
    """
    Create a custom CNN for FashionMNIST.

    The default is a really, awful design
    """
    def __init__(self):
        # initialize nn.Module
        super(MyCNN, self).__init__()
        # A layer of 3x3 conv with 32 filters, output will be same heigh/width as input (which are 28x28)
        # output will be of shape (batch size, 32, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        # downsampling layer using 4x4 max pooling, giving us an output of shape (batch size, 32, 7, 7)
        self.pool = nn.MaxPool2d(4, 4)
        self.resblock = ResidualBlock(32, 32)
        # fully connected layer connecting output of pool to next dense layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # we will use ReLU for all activations (except output layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward method for network
        :param x: input, for fashion MNIST, this is of shape (batch size, 28, 28)
        :return: output logits, shape (batch size, 10)
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.resblock(x)
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, trainloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")


# Train model
train_model(model, trainloader, criterion, optimizer, epochs=10)

# Evaluation function
def evaluate_model(model, testloader):
    model.eval()
    correct, total = 0, 0
    all_predictions = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return all_predictions


# Evaluate model
predictions = evaluate_model(model, testloader)


# Save predictions for submission
import json
submission_data = {"name": "Student_Name", "predictions": predictions}
with open("submission.json", "w") as f:
    json.dump(submission_data, f)

# Save trained model
torch.save(model, "model.pth")

print(predictions)
print(len(predictions))


import requests
# Send results

data = {
    "name": "G Money",
    "predictions": predictions
}
response = requests.post("https://csc7700leaderboard-d4fce9d9h2b5h8ab.centralus-01.azurewebsites.net/submit", json=data)
print(response.json())