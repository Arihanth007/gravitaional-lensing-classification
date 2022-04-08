import torch
import cv2
import glob
import argparse
import numpy as np
from models import EquiNet
from dataloaders import LensingData
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data():

    sub_images = np.array([np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
                           for file in glob.glob('sub/*.jpg')])
    sub_label = np.ones(len(sub_images), dtype=np.int32)

    nosub_images = np.array([np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
                            for file in glob.glob('no_sub/*.jpg')])
    nosub_label = np.zeros(len(nosub_images), dtype=np.int32)

    images = np.concatenate((sub_images, nosub_images))
    labels = np.concatenate((sub_label, nosub_label))

    my_data = []
    for i in range(images.shape[0]):
        my_data.append((images[i], labels[i]))

    my_data = np.array(my_data, dtype=object)
    np.random.shuffle(my_data)

    train_size = int(0.8 * len(my_data))
    test_size = len(my_data) - train_size
    batch_size = 32

    train_data = LensingData(data=my_data[:train_size],
                             transform=transformer)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size)

    test_data = LensingData(data=my_data[train_size:],
                            transform=transformer)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


transformer = T.Compose([
    T.ToTensor(),
    T.Resize(28),
])


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = torch.nn.BCELoss()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())

        loss = criterion(output.squeeze(-1), target.float())
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            train_auc = roc_auc_score(target.detach().to(
                'cpu'), output.detach().to('cpu'))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} AUC: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), train_auc))


def test(model, device, test_loader, rotate):
    model.eval()
    test_loss = 0
    correct = 0
    k = 0
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if rotate:
                data = torch.rot90(data, k, (2, 3))
                k += 1

            output = model(data.float())
            test_auc = roc_auc_score(target.detach().to(
                'cpu'), output.detach().to('cpu'))
            test_loss += criterion(output.squeeze(-1), target.float())
            pred = torch.round(output)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    title = ("Test rotated" if rotate else "Test upright")
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) AUC: {}\n'.format(
        title,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), test_auc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # print(f"Random seed: {torch.random.initial_seed()}")
    good_seeds = [9500892654245870846, 12718285161054196198]
    torch.random.manual_seed(good_seeds[0])

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = EquiNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.gamma)

    train_loader, test_loader = load_data()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, False)
        test(model, device, test_loader, True)
        scheduler.step()

    torch.save(model.state_dict(), "ecnn.pth")
    if args.save_model:
        torch.save(model.state_dict(), "ecnn.pth")


if __name__ == '__main__':
    main()
