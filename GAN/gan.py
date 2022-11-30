import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use('Agg') 
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math


class Generator(nn.Module):
    def __init__(self, dataset, latent_space_dm):
            super(Generator, self).__init__()
            self.dataset = dataset
            if dataset == "MNIST":
                self.main = nn.Sequential(
                    nn.Linear(latent_space_dm, 256),
                    nn.LeakyReLU(0.2),

                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2),

                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.2),

                    nn.Linear(1024, 784),
                    nn.Tanh(),
                )
            else:
                self.main = nn.Sequential(
                    nn.ConvTranspose2d( latent_space_dm, out_channels_g * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels_g * 8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(out_channels_g * 8, out_channels_g * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels_g * 4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( out_channels_g * 4, out_channels_g * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels_g * 2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( out_channels_g * 2, in_channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                )



    def forward(self, x):
        if self.dataset=="MNIST":
           return self.main(x).view(-1, 1, 28, 28)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self,dataset):
        super(Discriminator, self).__init__()
        self.dataset = dataset
        if self.dataset == "MNIST":
            self.main = nn.Sequential(
                    nn.Linear(784, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(256, 1),
                    nn.Sigmoid(),

            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels_d, out_channels_d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels_d * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels_d * 2, out_channels_d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels_d * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels_d * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )


    def forward(self, x):
        if self.dataset == "MNIST":
            x = x.view(-1, 784)
            return self.main(x)
        return self.main(x) 



def main(dataset, latent_space_dm):
    
    if dataset == "CIFAR10":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5, 0.5),(0.5, 0.5, 0.5))
                ])
        train_dataset = torchvision.datasets.CIFAR10(root='../DATA_CIFAR10',train=True, transform=transform, download=True)
    elif dataset == "MNIST":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,),(0.5,)),
                ])
        train_dataset = torchvision.datasets.MNIST(root='../DATA_MNIST',train=True, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


    generator = Generator(dataset,latent_space_dm).to(device)
    if dataset != "MNIST":
        generator.apply(weights_init)
    discriminator = Discriminator(dataset).to(device)
    if dataset != "MNIST":
        discriminator.apply(weights_init)

    optim_g = optim.Adam(generator.parameters(), lr=learning_rate,  betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=learning_rate,  betas=(0.5, 0.999))


    criterion = nn.BCELoss()


    losses_g = [] 
    losses_d = [] 
    images = []

    epoch_list = []

    jsd_losses_g = []
    jsd_losses_d = []

    fixed_noise = create_noise(sample_size, latent_space_dm) if dataset=="MNIST" else torch.randn(64, latent_space_dm, 1, 1, device=device)



    for epoch in range(epochs):
        loss_g = 0.0
        loss_d = 0.0

        jsd_loss_d = 0.0
        jsd_loss_g = 0.0

        for bi, data in tqdm(enumerate(train_loader), total = int(len(train_dataset) / train_loader.batch_size)):
            image, _ = data
            image = image.to(device)
            b_size = len(image)

            
            for step in range(k):
               
                data_fake = generator(create_noise(b_size, latent_space_dm)).detach()
                data_real = image
                
                loss_d1= train_discriminator(discriminator, criterion, optim_d, data_real, data_fake)
                loss_d += loss_d1
                jsd_loss_d += 0.5 * (-loss_d1 + math.log(4))


            data_fake = generator(create_noise(b_size, latent_space_dm))
            loss_g1 = train_generator(generator, discriminator, criterion, optim_g, data_fake)
            loss_g += loss_g1


        generated_img = generator(fixed_noise).cpu().detach()
        generated_img = make_grid(generated_img, padding =2, normalize=True)
        save_generator_image(generated_img, f"plots/{dataset}/output/{latent_space_dm}_model_epoch{epoch+1}.png")
        images.append(generated_img)
        epoch_loss_g = loss_g / bi
        epoch_loss_d = loss_d / bi

        epoch_jsd_loss_d = (jsd_loss_d / bi).item()

        losses_g.append(epoch_loss_g.item())
        losses_d.append(epoch_loss_d.item())
        epoch_list.append(epoch + 1)

        jsd_losses_d.append(epoch_jsd_loss_d)
        epoch_str = f"Epoch {epoch + 1} of {epochs}"
        g_loss_str = f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}"
        d_loss_str = f"Discriminator JSD loss:  {epoch_jsd_loss_d}"
        print(epoch_str)
        print(g_loss_str)
        print(d_loss_str)
        logging.info(epoch_str)
        logging.info(g_loss_str)
        logging.info(d_loss_str)
        


    fig = plt.figure()
    plt.plot(epoch_list, losses_d, color='black', label = "discriminator loss")
    plt.plot(epoch_list, losses_g, color='red', label = "generator loss")
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('GAN {dataset} BCE (Binary Cross Entropy) loss {latent_space_dm}')
    plt.savefig(f"plots/{dataset}/gan_{latent_space_dm}_bce.png")
    plt.close()


    fig = plt.figure()
    plt.plot(epoch_list, jsd_losses_d, color='black')
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0, 1.05, step=0.1)) 
    plt.ylabel('JS Divergence')
    plt.title('GAN_{dataset}_JSD_{latent_space_dm}')
    plt.savefig(f"plots/{dataset}/gan_{latent_space_dm}_jsd.png")
    plt.close()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_discriminator(discriminator, criterion, optimizer, data_real, data_fake):
    discriminator.train()
    b_size = data_real.size(0) 
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)


    optimizer.zero_grad()

    output_real = discriminator(data_real)
    if dataset != "MNIST":
        output_real=output_real.view(-1)
    loss_real = criterion(output_real, real_label)
    

    output_fake = discriminator(data_fake) 
    if dataset != "MNIST":
        output_fake=output_fake.view(-1)
    loss_fake = criterion(output_fake, fake_label) 

    loss_real.backward()
    loss_fake.backward()
    optimizer.step()

    return loss_real+loss_fake


def train_generator(generator, discriminator, criterion, optimizer, data_fake):
    generator.train()
    b_size = data_fake.size(0)
    real_label = label_real(b_size) 

    optimizer.zero_grad()
    
    output = discriminator(data_fake)
    if dataset != "MNIST":
        output=output.view(-1)
    loss = criterion(output, real_label)
    

    loss.backward()
    optimizer.step()

    return loss


def label_real(size):
    if dataset=="MNIST":
        data = torch.ones(size,1)
        return data.to(device)

    real_label = 1.
    data = torch.full((size,), real_label, dtype=torch.float, device=device)
    return data

def label_fake(size):
    if dataset=="MNIST":
        data = torch.zeros(size,1)
        return data.to(device)
    fake_label = 0.
    data = torch.full((size,), fake_label, dtype=torch.float, device=device)
    return data

def create_noise(sample_size, latent_space_dm):
    if dataset == "MNIST": 
        return torch.randn(sample_size, latent_space_dm).to(device)
    return torch.randn(sample_size, latent_space_dm, 1, 1, device=device)

def save_generator_image(image, path):
    save_image(image, path)

if __name__=="__main__":
      
    SEED = 786
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set Variables
    batch_size = 128
    image_size = 32
    in_channels = 3 
    latent_space_dms = [64,128]
    out_channels_g = 32
    out_channels_d = 32
    epochs = 50
    learning_rate=2e-4
    sample_size = 64
    k = 1
    datasets = [ "MNIST", "CIFAR10"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ",device)

    for dataset in datasets:
        for latent_space_dm in latent_space_dms:
            print("\n\n\nNEW RUN: ",dataset," ",latent_space_dm)
            logfilename=f"../logs/{dataset}_GAN_{latent_space_dm}.log"
            f = open(logfilename, "w")
            f.close()
            logging.basicConfig(filename=logfilename,level=logging.INFO)
            main(dataset, latent_space_dm)
