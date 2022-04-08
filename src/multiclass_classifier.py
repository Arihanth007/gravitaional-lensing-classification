import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from models import MultiClassImageClassifier
from dataloaders import CustomImageDataset
from sklearn.metrics import roc_auc_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data(location):

    data = CustomImageDataset(location+'train', ['sphere', 'no', 'vort'])

    train_data, test_data = torch.utils.data.random_split(data, [22500, 7500])
    val_data = CustomImageDataset(location+'val', ['sphere', 'no', 'vort'])

    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=128, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, shuffle=True,
                             batch_size=128, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, shuffle=True,
                            batch_size=128, num_workers=4, pin_memory=True)

    return [train_loader, test_loader, val_loader], [train_data, test_data, val_data]


def test(model, criterion, data_loader, data, eval):
    acc = 0
    epoch_loss = []
    model.eval()

    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)

            out = model(X.float())
            loss = criterion(out, y.long())
            epoch_loss.append(loss.detach().to('cpu').item())

            acc += torch.sum(out.argmax(dim=1) == y)
            auc = roc_auc_score(
                y.detach().to('cpu'), out.detach().to('cpu'), multi_class='ovr')

    acc = acc/len(data)
    print(f'{eval} Loss: {np.mean(epoch_loss)} {eval} Accuracy: {acc}')
    print(f'{eval} AUC: {auc}\n')

    return acc


def train(model, criterion, optimizer, loaders, data, loc, train_epochs=20):

    train_loader, test_loader, val_loader = loaders
    train_data, test_data, val_data = data

    best_accuracy = 0

    for epoch in range(train_epochs):

        epoch_loss = []
        acc = 0
        model.train()

        for _, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            out = model(X.float())
            loss = criterion(out, y.long())
            loss.backward()
            optimizer.step()

            train_auc = roc_auc_score(
                y.detach().to('cpu'), out.detach().to('cpu'), multi_class='ovr')
            epoch_loss.append(loss.detach().to('cpu').item())
            acc += torch.sum(out.argmax(dim=1) == y)

        print(f'Epoch-{epoch}: {acc}/{len(train_data)} {acc/len(train_data)}')
        print(f'AUC: {train_auc}')
        print(f'Train Loss: {np.mean(epoch_loss)}\n')

        val_acc = test(model, criterion, val_loader, val_data, 'val')
        val_acc = test(model, criterion, test_loader, test_data, 'test')

        if val_acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), f'{loc}model.pth')

        print(f'Best Accuracy of the model is {best_accuracy}\n')


if __name__ == '__main__':

    on_device = '/Users/arihanthtadanki/Downloads/dataset/'
    on_ada = '/scratch/arihanth.srikar/dataset/'

    loaders, data = get_data(on_ada)
    train_loader, test_loader, val_loader = loaders
    train_data, test_data, val_data = data

    model = MultiClassImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    train(model, criterion, optimizer, loaders, data, on_ada)
