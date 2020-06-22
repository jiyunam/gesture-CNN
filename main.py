'''
    The entry into your code. This file should include a training function and an evaluation function.
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import os

from torch.utils.data import DataLoader
from dataset import GestureDataset
from sklearn.preprocessing import LabelEncoder
from model import CNN
from train_val_split import data_train, data_valid, label_train, label_valid

np.random.seed(100)
torch.manual_seed(100)
seed = 100

def load_data(batch_size):
    train_dataset = GestureDataset(data_train, label_train)
    valid_dataset = GestureDataset(data_valid, label_valid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(lr):
    loss_fnc = torch.nn.CrossEntropyLoss()
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set."""

    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    label_encoder = LabelEncoder()

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.permute(0, 2, 1)
        labels = np.asarray(labels)
        labels = label_encoder.fit_transform(labels)
        labels = torch.from_numpy(labels)

        outputs = net(inputs)
        loss = criterion(input=outputs, target=labels)
        outputs = outputs.detach().numpy()
        predictions = outputs.argmax(axis=1)
        corr = predictions != labels

        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)

    return err, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    MaxEpochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size

    net, loss_fnc, optimizer = load_model(lr=lr)
    train_loader, val_loader = load_data(batch_size)

    train_err = np.zeros(MaxEpochs)
    train_loss = np.zeros(MaxEpochs)
    val_err = np.zeros(MaxEpochs)
    val_loss = np.zeros(MaxEpochs)

    # Train the network
    label_encoder = LabelEncoder()
    epoch_arr = []

    for epoch in range(MaxEpochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        epoch_arr.append(epoch)

        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs = inputs.permute(0,2,1)
            labels = np.asarray(labels)
            labels = label_encoder.fit_transform(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            labels = torch.from_numpy(labels)

            loss = loss_fnc(input=outputs, target=labels)
            loss.backward()
            optimizer.step()

            outputs = outputs.detach().numpy()
            predictions = outputs.argmax(axis=1)

            # Calculate the statistics
            corr = predictions != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, loss_fnc)
        print("Epoch {} | Train acc: {} | Validation acc: {}".format(epoch + 1, 1 - train_err[epoch], 1-val_err[epoch]))


    print('Finished Training')
    print("max val acc: ", 1-min(val_err))
    print("epoch: %s, lr: %s, batchsize: %s" %(MaxEpochs, lr, batch_size))
    best_val = 1 - min(val_err)

    # Plot Validation and Training Data
    plt.figure()
    plt.title("Validation and Training Accuracy (Best Validation Accuracy: %s)" %best_val)
    plt.plot(epoch_arr, 1-train_err, label="Training")
    plt.plot(epoch_arr, 1-val_err, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("Val_Train_bs%s_lr%s_epoch%s_bestval%s.png" %(batch_size,lr,MaxEpochs,best_val))


    # Save Model
    torch.save(net, "model.pt")


    # Run test dataset
    test_data = np.load('assign3part3/test_data.npy')
    data_test = np.empty((1170, 100, 6))
    for i in range(len(test_data)):
        vec = test_data[i, :, :]
        for j in range(100):
            data_test[i, j, :] = np.array([np.divide((vec[j] - np.mean(vec, axis=0)), np.std(vec, axis=0))])

    model = torch.load("model.pt")

    labels = np.random.rand(1170, 26)
    test_dataset = GestureDataset(data_test, labels)
    test_loader = DataLoader(test_dataset, batch_size=1170, shuffle=False)

    predictions = np.zeros((1170, 1))
    for i, batch in enumerate(test_loader):
        inputs, labels = batch
        inputs = np.einsum('kli->kil', inputs)
        inputs = torch.from_numpy(inputs)
        output_test = model(inputs)
        for i in range(1170):
            predictions[i] = torch.argmax(output_test[i])

    np.savetxt("predictions.txt", predictions.squeeze())


if __name__ == "__main__":
    main()