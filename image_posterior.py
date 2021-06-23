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
from naive_ae import ConvAutoencoder

DATA_PATH = '../data_sets/mnist'
NAIVE_AE_PATH = './trained_models/convAutoEncSigmoid/naive_ae25.pth'
CLEVER_AE_PATH = './trained_models/convAutoEncNoSigmoid/naive_ae25.pth'


def posterior_loss_denoising(I, I_c, AE, sigma, T):
    likelihood_term = torch.exp(-torch.norm(I - I_c)) / 2 * (sigma**2)
    prior_term = torch.norm(AE(I_c) - I) / T
    # print(f'likelyhood_term:{likelihood_term} prior_term:{prior_term}')
    # print(f'loss: {-torch.log(likelihood_term)}, { - torch.log(prior_term)}')
    return -torch.log(likelihood_term) - torch.log(prior_term)


def maximize_posterior_denoising(I_c, AE, sigma=1, T=0.1):
    I_0 = torch.rand(1,1,28, 28, requires_grad=True)
    
    I_i = I_0
    optimizer = torch.optim.Adam([I_i], lr=0.1)
    for i in range(2000):
        loss = posterior_loss_denoising(I_i, I_c, AE, sigma, T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return I_i


def posterior_loss_mid_suppression(I, I_c, AE, T):
    # I = suppress_mid(I)
    prior_term = torch.norm(AE(I_c) - I) / T
    return - torch.log(prior_term)


def maximize_posterior_mid_suppression(I_c, AE, sigma=1, T=100):
    I_0 = torch.rand(1,1,28, 28, requires_grad=True)
    
    I_i = I_0
    optimizer = torch.optim.Adam([I_i], lr=0.1)
    for i in range(2000):
        loss = posterior_loss_mid_suppression(I_i, I_c, AE, T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return I_i

def gaussian_noise(I):
    return I + torch.randn(1,1,28,28)

def suppress_mid(I):
    I_c = torch.clone(I)
    I_c[:,:,9:18,9:18] = 0
    return I_c

naive_ae = ConvAutoencoder()
naive_ae.load_state_dict(torch.load(NAIVE_AE_PATH))

clever_ae = ConvAutoencoder()
clever_ae.load_state_dict(torch.load(CLEVER_AE_PATH))


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_set = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)


############################
#      denoising task      #
############################
I = test_set[2][0]
I_c = gaussian_noise(I)

naive_denoising = maximize_posterior_denoising(I_c, naive_ae)
clever_denoising = maximize_posterior_denoising(I_c, clever_ae)

fig, ax = plt.subplots(2,2)
fig.suptitle('denoising task')

ax[0,0].imshow(I.squeeze())
ax[0,0].set_title('original image')

ax[0,1].imshow(I_c.squeeze())
ax[0,1].set_title('noised image')

ax[1,0].imshow(naive_denoising.detach().squeeze())
ax[1,0].set_title('naive AE denoising')

ax[1,1].imshow(clever_denoising.detach().squeeze())
ax[1,1].set_title('clever AE denoising')


############################
#      inpainting task     #
############################
I = test_set[2][0].view(1,1,28,28)
I_c = suppress_mid(I)

naive_inpainting = maximize_posterior_mid_suppression(I_c, naive_ae)
clever_inpainting = maximize_posterior_mid_suppression(I_c, clever_ae)

fig, ax = plt.subplots(2,2)
fig.suptitle('inpainting task')

ax[0,0].imshow(I.squeeze())
ax[0,0].set_title('original image')

ax[0,1].imshow(I_c.squeeze())
ax[0,1].set_title('noised image')

ax[1,0].imshow(naive_inpainting.detach().squeeze())
ax[1,0].set_title('naive AE inpainting')

ax[1,1].imshow(clever_inpainting.detach().squeeze())
ax[1,1].set_title('clever AE inpainting')


plt.show()