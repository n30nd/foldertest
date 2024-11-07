import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import copy 
from tqdm import tqdm
from dataset import prepare_dataset_for_centralized_train
# Note the model and functions here defined do not have any FL-specific components.


# class Net(nn.Module):
#     """A simple CNN suitable for simple vision tasks."""

#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

import torch.nn as nn

class Net(nn.Module):
    """Simplified CNN for demonstration purposes."""
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str, type_fed = 'FedAvg', proximal_mu: float = 0.1):
    """Train the network on the training set.
    This is a fairly simple training loop for PyTorch.
    """
    if type_fed == 'FedAvg':
        criterion = torch.nn.CrossEntropyLoss()
        net.train()
        net.to(device)
        # for _ in range(epochs):
        #     for images, labels in trainloader:
        #         images, labels = images.to(device), labels.to(device)
        #         optimizer.zero_grad()
        #         loss = criterion(net(images), labels)
        #         loss.backward()
        #         optimizer.step()
        for epoch in tqdm(range(epochs), desc="Training FedAvg"):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()

    if type_fed == 'FedProx':
        global_params = copy.deepcopy(net).parameters()
        criterion = torch.nn.CrossEntropyLoss()
        net.train()
        net.to(device)
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                proximal_term = torch.tensor(0.0, device=device)
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                loss = criterion(net(images), labels) + proximal_mu / 2 * proximal_term
                loss.backward()
                optimizer.step()

def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    print("Testing...")
    y_true, y_pred = [], []
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    loss /= len(testloader.dataset)
    # accuracy = correct / len(testloader.dataset)
    return loss, accuracy

#Huan luyen tap trung
def centralize_training(device, cfg ):
    epochs = cfg.num_rounds
    lr = cfg.config_fit.lr

    model = Net()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    trainloader, valloader, testloader = prepare_dataset_for_centralized_train(cfg.batch_size)
    
    #CENTRALIZED TRAINING == CT
    loss_val_CT = []
    acc_val_CT = []
    loss_test_CT = []
    acc_test_CT = []

    for _ in range(epochs):
        train(model, trainloader, optim, 1, device, type_fed = 'FedAvg')
        loss, acc = test(model, valloader, device)
        loss_val_CT.append(loss)
        acc_val_CT.append(acc)
        loss, acc = test(model, testloader, device)
        loss_test_CT.append(loss)
        acc_test_CT.append(acc)

    return loss_val_CT, acc_val_CT, loss_test_CT, acc_test_CT
