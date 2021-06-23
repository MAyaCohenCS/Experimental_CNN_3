# based on:
# https://analyticsindiamag.com/hands-on-guide-to-implement-deep-autoencoder-in-pytorch-for-image-reconstruction/

import os
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import pylab as py
from IPython import embed

DATA_PATH = '../data_sets/mnist'
RESULTS_PATH = './results/naive_out'




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256) # Input image (28*28 = 784)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        #Decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784) # Output image (28*28 = 784)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        return x

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))

        return x

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)

        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        #Encoder
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5) 
        self.conv2 = nn.Conv2d(4 ,6,kernel_size=5)
        self.conv3 = nn.Conv2d(6 ,8,kernel_size=5)
        self.enc_fc = nn.Linear(in_features=2048, out_features=8)

        #Decoder 
        self.dec_fc = nn.Linear(in_features=8, out_features=2048)
        self.deconv3 = nn.ConvTranspose2d(8,6,kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(6,4,kernel_size=5)
        self.deconv1 = nn.ConvTranspose2d(4,1,kernel_size=5)


    def encode(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, 2048)
        x = torch.sigmoid(self.enc_fc(x))
        # x = F.relu(self.enc_fc(x))

        return x

    def decode(self, x):
        x = F.relu(self.dec_fc(x))
        x = x.view(-1, 8, 16,16)
        x = F.leaky_relu(self.deconv3(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.leaky_relu(self.deconv1(x))

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def make_dir():
    image_dir = RESULTS_PATH
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def save_decod_img(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, os.path.join(RESULTS_PATH, 'Autoencoder_image{}.png'.format(epoch)))


def training(model, train_loader, Epochs):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            # img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss))

        if epoch % 5 == 0:
            save_decod_img(outputs.cpu().data, epoch)
            torch.save(model.state_dict(), f'trained_models/naive_ae{epoch}.pth')

    
    return train_loss

def test_image_reconstruct(model, test_loader):
    for batch in test_loader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = model(img)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, os.path.join(RESULTS_PATH,'MNIST_reconstruction.png'))
        break


if __name__ == '__main__':
        
    Epochs = 30
    Lr_Rate = 1e-3
    batch_size = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    train_set = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


    model = ConvAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)



    device = get_device()
    model.to(device)
    make_dir()


    # train_loss = training(model, train_loader, Epochs)

    # plt.figure()
    # plt.plot(train_loss)
    # plt.title('Train Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.savefig('deep_ae_mnist_loss.png')

    model.load_state_dict(torch.load(f'trained_models/convAutoEncSigmoid/naive_ae25.pth'))

    imgs_pack = torch.zeros(0, 1, 28, 28).to(device)
    label_pack = torch.zeros(0)
    codes_pack = torch.zeros(0, 8).to(device).to(device)

    sample_counter = 0

    for batch in test_loader:
        imgs, labels = batch
        imgs = imgs.to(device)
        imgs_pack = torch.cat([imgs_pack, imgs])
        codes = model.encode(imgs)

        codes_pack = torch.cat([codes_pack, codes])
        label_pack = torch.cat([label_pack, labels])

        sample_counter += len(labels)
        if sample_counter >= 300:
            break

    codes_pack = codes_pack.cpu()

    def draw_subspace(codes, dim_x, dim_y, labels):
        plt.figure()

        cmap = plt.get_cmap('viridis')
        for i in range(0,10):
            plt.scatter(codes[np.where(labels==i),dim_x],codes[np.where(labels==i), dim_y], color=cmap(i/9), marker='$'+str(int(i))+'$')


        label_num = 11
        lable_name = 'X'
        plt.scatter(codes[np.where(labels==label_num),dim_x],codes[np.where(labels==label_num), dim_y], color='#800000', marker='$'+lable_name+'$')
        
        label_num = 12
        lable_name = 'S'
        plt.scatter(codes[np.where(labels==label_num),dim_x],codes[np.where(labels==label_num), dim_y], color='#400000', marker='$'+lable_name+'$')

        plt.title(f' latent space: dims:{dim_x},{dim_y}')
        plt.savefig(f'results/subspace_slieces/{dim_x}_{dim_y}.png')
        
        plt.close()


    dim_pairs = [(i,j) for i in range(8) for j in range(i+1,8)]
    for i,j in dim_pairs:
        draw_subspace(np.array(codes_pack.detach()), i,j, np.array(label_pack))    

