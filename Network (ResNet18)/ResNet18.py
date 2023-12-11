# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

# Define the Residual Block class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for identity mapping
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    # Forward pass for the Residual Block
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)  # Add the shortcut connection
        out = self.relu(out)

        return out

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the four ResNet layers
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    # Helper function to create a ResNet layer
    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    # Forward pass for the ResNet model
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Custom dataset class for loading data
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Use torchvision's ImageFolder to load images from the specified path
        self.data = datasets.ImageFolder(root=data_path, transform=transform)
        self.classes = self.data.classes  # Save the classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # img is the input data (features)
        # label is the ground truth label
        return img, label

# Training function
def train(model, train_loader, criterion, optimizer, device):
    """
    Trains the given PyTorch model using the specified DataLoader for training data.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - train_loader (DataLoader): DataLoader providing batches of training data.
    - criterion (nn.Module): The loss function used for optimization.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - device (torch.device): The device on which the training is performed (e.g., "cuda" or "cpu").

    Returns:
    - float: The average training loss over all batches in the training DataLoader.
    """
    # Set the model to training mode
    model.train()
    
    # Initialize running loss for the current epoch
    running_loss = 0.0

    # Iterate through batches in the training DataLoader
    for inputs, labels in tqdm(train_loader, desc="Training"):
        # Move input data and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients in the optimizer
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)

        # Calculate the batch loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the model's parameters using the optimizer
        optimizer.step()

        # Update the running loss for the current epoch
        running_loss += loss.item()

    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)

    # Return the average training loss
    return avg_train_loss


# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

# Hyperparameters
num_epochs = 6
learning_rate = 0.00001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of ResNet
resnet = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=2).to(device)

# Create DataLoaders for training, validation, and testing data
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(data_path=r'DL2\train', transform=data_transform)
test_dataset = CustomDataset(data_path=r'DL2\test', transform=data_transform)
test_dataset2 = CustomDataset(data_path=r'DL2\test2', transform=data_transform)#Only used for visualizing the model

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=5, shuffle=False)

# Get the classes from the train_loader
classes = train_loader.dataset.classes

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

# Initialize empty lists to store training and test metrics
train_loss_history = []
test_loss_history = []
accuracy_history = []

# Training loop with validation
for epoch in range(num_epochs):
    # Train the model
    avg_train_loss = train(resnet, train_loader, criterion, optimizer, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Test the model
    test_loss, test_accuracy = test(resnet, test_loader, criterion, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy * 100:.2f}%")

    # Store metrics for plotting
    train_loss_history.append(avg_train_loss)
    test_loss_history.append(test_loss)
    accuracy_history.append(test_accuracy)

# Save the trained model
torch.save(resnet.state_dict(), 'resnet_model.pth')

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize the predictions of the model on a subset of test data
def visualize_model(model, test_loader, criterion, num_images=30):
    # Save the current training mode
    was_training = model.training
    # Set the model to evaluation mode
    model.eval()
    
    # Variables to keep track of visualization statistics
    images_so_far = 0
    correct_predictions = 0
    total_samples = 0

    # Iterate through the test_loader to get input images and labels
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the device (CPU or GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Iterate through the batch of images
            for j in range(inputs.size()[0]):
                # Update the count of processed images
                images_so_far += 1
                
                # Create a subplot for each image
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                
                # Display the predicted and true labels as titles
                ax.set_title('predicted: {} truth: {}'.format(preds[j].item(), labels[j].item()))
                
                # Convert the image tensor to NumPy array and transpose the channels
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                # Display the image
                ax.imshow(img)

                # Update total sample count and correct predictions count
                total_samples += 1
                correct_predictions += (preds[j] == labels[j]).item()

                # Break the loop if the desired number of images is reached
                if images_so_far == num_images:
                    # Set the model back to its original training mode
                    model.train(mode=was_training)
                    # Calculate accuracy based on displayed predictions
                    accuracy = correct_predictions / total_samples
                    print(f'Test Accuracy: {accuracy * 100:.2f}%')
                    return
                    
        # Set the model back to its original training mode after visualization
        model.train(mode=was_training)

# Call the visualization function
visualize_model(resnet, test_loader2, criterion, num_images=30)

# Display the visualizations
plt.show()

# Print a summary of the model architecture
summary(resnet, input_size=(3, 224, 224))
