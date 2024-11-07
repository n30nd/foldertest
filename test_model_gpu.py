import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from dataset import prepare_dataset_for_centralized_train
from model import Net, train, test
from tqdm import tqdm
import time

import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with FedAvg or FedProx")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()


args = parse_args()
epochs = args.epochs
lr = args.lr
print('epochs = ', epochs)
print('lr = ', lr)
model = Net()
optim = torch.optim.Adam(model.parameters(), lr=lr)
trainloader, valloader, testloader = prepare_dataset_for_centralized_train(32)

#CENTRALIZED TRAINING == CT
loss_val_CT = []
acc_val_CT = []
loss_test_CT = []
acc_test_CT = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)
start_time = time.time()

for _ in range(epochs):
    train(model, trainloader, optim, 2, device, type_fed='FedAvg')
    loss, acc = test(model, valloader, device)
    loss_val_CT.append(loss)
    acc_val_CT.append(acc)
    loss, acc = test(model, testloader, device)
    loss_test_CT.append(loss)
    acc_test_CT.append(acc)

print(f'Loss_VAL_CT = {loss_val_CT}, Acc_VAL_CT = {acc_val_CT}, Loss_TEST_CT = {loss_test_CT}, Acc_TEST_CT = {acc_test_CT}, device = {device}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Elapsed time: {elapsed_time} seconds')