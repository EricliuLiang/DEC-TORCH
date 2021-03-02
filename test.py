import os
import time
import pdb
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from metrics import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
from procesess import DealDataset
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 28 * 28))
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class DEC(nn.Module):
    def __init__(self, n_clusters=10, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x)
        return self.clusteringlayer(x)

    def visualize(self, epoch, x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach()
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        fig.savefig('plots/mnist_{}.png'.format(epoch))
        plt.close(fig)


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, data):
        self.train_set = data
        # self.train_labels = train_labels

        self.data = np.vstack(self.train_set).reshape(-1, 28, 28)
        print(self.data.shape)
        # self.data = self.data.transpose((0, 1, 2))  # convert to HWC
        # print(self.data.shape)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(0.2),
            # transforms.GaussianBlur(kernel_size=int(self.args.img_size * 0.1), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):

        img1 = self.data[index]
        img2 = img1.copy()
        img1 = Image.fromarray(np.uint8(img1), mode='L')
        img1 = self.test_transform(img1)
        img1 = img1.squeeze()
        img1 = img1.view(-1, 1)
        img1 = img1.squeeze()
        # img2 = Image.fromarray(np.uint8(img2))
        # img2 = self.transform(img2)
        return img1, img2

    def __len__(self):
        return len(self.data)

def pretrain(**kwargs):
    data = kwargs['data']
    Y = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    parameters = list(autoencoder.parameters())
    optimizer = torch.optim.SGD(parameters, lr=1, momentum=0.9)
    dataset = DealDataset(data)
    train_loader = DataLoader(dataset=data,
                              batch_size=256,
                              shuffle=False,
                              num_workers=2)

    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            data = data.float()
            # print(data.shape)
            img = data.to(device)
            # img = img.view(img.size[0], -1)
            # ===================forward=====================
            output = model(img)
            output = output.squeeze(1)
            output = output.view(output.size(0), 28 * 28)
            loss = nn.MSELoss()(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, savepath,
            is_best)


def train(**kwargs):
    data = kwargs['data']
    labels = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    features = []
    train_loader = DataLoader(dataset=data,
                              batch_size=256,
                              shuffle=False)

    for i, batch in enumerate(train_loader):
        img = batch.float()
        img = img.to(device)
        features.append(model.autoencoder.encode(img).detach().cpu())
    features = torch.cat(features)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=10, n_init=20).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    y_pred = kmeans.predict(features)
    accuracy = acc(y.cpu().numpy(), y_pred)
    print('Initial Accuracy: {}'.format(accuracy))

    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    print('Training')
    row = []
    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float()
        img = img.to(device)
        output = model(img)
        target = model.target_distribution(output).detach()
        out = output.argmax(1)
        # if epoch % 20 == 0:
        #     print('plotting')
        #     dec.visualize(epoch, img)
        loss = loss_function(output.log(), target) / output.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
        row.append([epoch, accuracy])
        print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch, num_epochs, accuracy, loss))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, savepath,
            is_best)

    df = pd.DataFrame(row, columns=['epochs', 'accuracy'])
    df.to_csv('log.csv')


def load_mnist():
    # the data, shuffled and split between train and test sets
    train = MNIST(root='./data/',
                  train=True,
                  transform=transforms.ToTensor(),
                  download=True)

    test = MNIST(root='./data/',
                 train=False,
                 transform=transforms.ToTensor())
    x_train, y_train = train.data, train.targets
    x_test, y_test = test.data, test.targets
    x = torch.cat((x_train, x_test), 0)
    y = torch.cat((y_train, y_test), 0)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--save_dir', default='saves')
    args = parser.parse_args()
    # print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size

    # trainDataset = DealDataset('mnist/', "train-images-idx3-ubyte.gz",
    #                            "train-labels-idx1-ubyte.gz")
    # train_loader = DataLoader(
    #     dataset=trainDataset,
    #     batch_size=256,  # 一个批次可以认为是一个包，每个包中含有100张图片
    #     shuffle=True,
    # )
    x, y = load_mnist()
    autoencoder = AutoEncoder().to(device)
    ae_save_path = 'saves/sim_autoencoder.pth'
    #
    if os.path.isfile(ae_save_path):
        print('Loading {}'.format(ae_save_path))
        checkpoint = torch.load(ae_save_path)
        autoencoder.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(ae_save_path))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }
    pretrain(data=x,labels=y, model=autoencoder, num_epochs=epochs_pre, savepath=ae_save_path, checkpoint=checkpoint)
    #
    dec_save_path = 'saves/dec.pth'
    dec = DEC(n_clusters=10, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)
    if os.path.isfile(dec_save_path):
        print('Loading {}'.format(dec_save_path))
        checkpoint = torch.load(dec_save_path)
        dec.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(dec_save_path))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }
    train(data=x, labels=y, model=dec, num_epochs=args.train_epochs, savepath=dec_save_path, checkpoint=checkpoint)