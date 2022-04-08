import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image

import torchvision
import groupcnn
import argparse
import cv2
import glob
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class MnistRotDataset(Dataset):

    def __init__(self, data, transform=None):

        self.transform = transform

        self.images = data[:, 0]
        self.labels = data[:, 1]
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


image_size = 150
image_pad_size = 151

# images are padded to have shape 29x29.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = T.Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = T.Resize(453)
resize2 = T.Resize(151)
totensor = T.ToTensor()


train_size = int(0.8 * len(images))
test_size = len(images) - train_size
batch_size = 32

train_transform = T.Compose([
    totensor,
    T.Resize(28),
])

test_transform = T.Compose([
    totensor,
    T.Resize(28),
])

mnist_train = MnistRotDataset(data=my_data[:train_size],
                              transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)


mnist_test = MnistRotDataset(data=my_data[train_size:],
                             transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

image, label = next(iter(train_loader))
print(f"\nImage size: {image.shape}\nLabel shape: {label.shape}")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = groupcnn.ConvZ2P4(1, 8, 5)
        self.pool1 = groupcnn.MaxSpatialPoolP4(2)
        self.conv2 = groupcnn.ConvP4(8, 32, 3)
        self.pool2 = groupcnn.MaxSpatialPoolP4(2)
        self.conv3 = groupcnn.ConvP4(32, 64, 3)
        self.pool3 = groupcnn.MaxSpatialPoolP4(2)
        self.conv4 = groupcnn.ConvP4(64, 32, 3)
        self.pool4 = groupcnn.MaxRotationPoolP4()
        self.pool5 = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(32, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.pool1(x))
        x = self.conv2(x)
        x = torch.nn.functional.relu(self.pool2(x))
        x = self.conv3(x)
        x = torch.nn.functional.relu(self.pool3(x))
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        # output = torch.nn.functional.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
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

            # for u in range(4):
                # data2 = torch.rot90(data, u, (2, 3))
                # output = model(data2)

            output = model(data.float())
            test_auc = roc_auc_score(target.detach().to(
                'cpu'), output.detach().to('cpu'))
            test_loss += criterion(output.squeeze(-1), target.float())
            # sum up batch loss
            # test_loss += torch.nn.functional.nll_loss(
            #     output.float(), target.long(), reduction='sum').item()
            # get the index of the max log-probability
            # pred = output.argmax(dim=1, keepdim=True)
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
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
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

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, False)
        test(model, device, test_loader, True)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_gcnn.pth")


if __name__ == '__main__':
    main()
