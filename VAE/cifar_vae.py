import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib
matplotlib.use('Agg') 
import random
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


SEED = 786
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



class VAE(nn.Module):
	def __init__(self,latent_space_dm):
		super(VAE, self).__init__()
		
		#Encoding layers
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(16)

		self.fc1 = nn.Linear(8 * 8 * 16, 512)
		self.fc_bn1 = nn.BatchNorm1d(512)
		self.fc21 = nn.Linear(512, 512)
		self.fc22 = nn.Linear(512, 512)


		#Decoding layers
		self.fc3 = nn.Linear(512, 512)
		self.fc_bn3 = nn.BatchNorm1d(512)
		self.fc4 = nn.Linear(512, 8 * 8 * 16)
		self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

		self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(32)
		self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn6 = nn.BatchNorm2d(32)
		self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn7 = nn.BatchNorm2d(16)
		self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

		self.relu = nn.ReLU()

	def encoder(self, x):
		conv1 = self.relu(self.bn1(self.conv1(x)))
		conv2 = self.relu(self.bn2(self.conv2(conv1)))
		conv3 = self.relu(self.bn3(self.conv3(conv2)))
		conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

		fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
		#mu, log_var
		return self.fc21(fc1), self.fc22(fc1)

	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5*log_var) 
		eps = torch.randn_like(std)
		sample = mu + eps * log_var
		return sample

	def decoder(self, z):
		fc3 = self.relu(self.fc_bn3(self.fc3(z)))
		fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

		conv5 = self.relu(self.bn5(self.conv5(fc4)))
		conv6 = self.relu(self.bn6(self.conv6(conv5)))
		conv7 = self.relu(self.bn7(self.conv7(conv6)))

		return self.conv8(conv7).view(-1, 3, 32, 32)

	def forward(self, x):
		mu, log_var = self.encoder(x)
		z = self.reparameterize(mu, log_var)
		return self.decoder(z), mu, log_var 



def main(latent_space_dm):


	train_dataset = torchvision.datasets.CIFAR10(root='./DATA_CIFAR10',train=True, transform=transforms.ToTensor(), download=True)
	test_dataset = torchvision.datasets.CIFAR10(root='./DATA_CIFAR10',train=False, transform=transforms.ToTensor(),download = True)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


	model = VAE(latent_space_dm).to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	criterion = nn.MSELoss(reduction='sum')

	train_losses = []
	train_losses_MSE = []
	train_losses_KLD = []

	test_losses = []
	test_losses_MSE = []
	test_losses_KLD = []

	epoch_list = []
	for epoch in range(EPOCHS):
		print(f"Epoch {epoch+1}:")
		train_epoch_loss_MSE, train_epoch_loss_KLD, train_epoch_loss = train(model, optimizer, train_loader, criterion)
		test_epoch_loss_MSE, test_epoch_loss_KLD, test_epoch_loss = evaluate(model, optimizer, test_loader, BATCH_SIZE, epoch, criterion,latent_space_dm)
		train_losses.append(train_epoch_loss)
		train_losses_MSE.append(train_epoch_loss_MSE)
		train_losses_KLD.append(train_epoch_loss_KLD)
		test_losses.append(test_epoch_loss)
		test_losses_MSE.append(test_epoch_loss_MSE)
		test_losses_KLD.append(test_epoch_loss_KLD)
		epoch_list.append(epoch+1)
		print(f"Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}, Test MSE Loss: {test_epoch_loss_MSE:.4f}, Test KLD Loss: {train_epoch_loss_KLD:.4f}")

	fig = plt.figure()
	plt.plot(epoch_list, train_losses, color='black', label = "train loss")
	plt.plot(epoch_list, test_losses, color='red', label = "test loss")
	plt.legend(loc='best', prop={'size': 10})
	plt.xlabel('Epoch')
	plt.ylabel('Reconstruction Loss and KL Divergence')
	plt.title('VAE_CIFAR10_Reconstruction_Loss_and_KL_Divergence')
	plt.savefig(f"plots/CIFAR10/vae_{latent_space_dm}_total_loss.png")
	plt.close()




	fig = plt.figure()
	plt.plot(epoch_list, train_losses_MSE, color='black', label = "train loss")
	plt.plot(epoch_list, test_losses_MSE, color='red', label = "test loss")
	plt.legend(loc='best', prop={'size': 10})
	plt.xlabel('Epoch')
	plt.ylabel('Mean Squared Error')
	plt.title('VAE_CIFAR10_Reconstruction_Loss')
	plt.savefig(f"plots/CIFAR10/vae_{latent_space_dm}_mse.png")
	plt.close()




	fig = plt.figure()
	plt.plot(epoch_list, train_losses_KLD, color='black', label = "train loss")
	plt.plot(epoch_list, test_losses_KLD, color='red', label = "test loss")
	plt.legend(loc='best', prop={'size': 10})
	plt.xlabel('Epoch')
	plt.ylabel('KL Divergence')
	plt.title('VAE_CIFAR10_KL_Divergence')
	plt.savefig(f"plots/CIFAR10/vae_{latent_space_dm}_kld.png")
	plt.close()



def loss_function(mse_loss, mu, log_var):
	MSE = mse_loss
	KLD = -0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())

	return MSE + KLD, KLD


def train(model, optimizer, train_loader, criterion):
	model.train()
	running_loss = 0.0
	running_loss_KLD = 0.0
	running_loss_MSE = 0.0
	for batch_idx, (images, _ ) in tqdm(enumerate(train_loader), total = int(len(train_loader.dataset) / BATCH_SIZE)):
			images = images.to(device)
			optimizer.zero_grad()
			recon_batch, mu, log_var = model(images)
			mse_loss = criterion(recon_batch, images)
			loss, kld_loss = loss_function(mse_loss, mu, log_var)
			loss.backward()
			running_loss += loss.item()
			running_loss_MSE += mse_loss.item()
			running_loss_KLD += kld_loss.item()
			optimizer.step()
	train_loss = running_loss/len(train_loader.dataset)
	train_loss_MSE = running_loss_MSE/len(train_loader.dataset)
	train_loss_KLD = running_loss_KLD/len(train_loader.dataset)
	return train_loss_MSE, train_loss_KLD, train_loss


def evaluate(model, optimizer, test_loader, batch_size, epoch, criterion, latent_space_dm):
	model.eval()
	running_loss = 0.0
	running_loss_KLD = 0.0
	running_loss_MSE = 0.0
	with torch.no_grad():
			for batch_idx, (images, _ ) in enumerate(test_loader):
					images = images.to(device)
					recon_batch, mu, log_var = model(images)
					mse_loss = criterion(recon_batch, images)
					loss, kld_loss = loss_function(mse_loss, mu, log_var)
					running_loss += loss.item()
					running_loss_MSE += mse_loss.item()
					running_loss_KLD += kld_loss.item()

					if batch_idx == int(len(test_loader.dataset)/batch_size) - 1:
						recon_batch_ = recon_batch.view(batch_size, 3, 32, 32)[:64]
						generated_img = make_grid(recon_batch_, padding =2, normalize = True)
						save_generator_image(generated_img, f"plots/CIFAR10/output/{latent_space_dm}_epoch{epoch+1}.png")

						if epoch == EPOCHS-1:
							image_batch = images.view(batch_size, 3, 32, 32)[:64]
							real_img = make_grid(image_batch, padding =2, normalize = True)
							save_generator_image(real_img, f"plots/CIFAR10/output/{latent_space_dm}_epoch{epoch+1}.png")

	test_loss = running_loss/len(test_loader.dataset)
	test_loss_MSE = running_loss_MSE/len(test_loader.dataset)
	test_loss_KLD = running_loss_KLD/len(test_loader.dataset)
	return test_loss_MSE, test_loss_KLD, test_loss

def save_generator_image(image, path):
	save_image(image, path)


if __name__=="__main__":
	  
	
	BATCH_SIZE = 128
	EPOCHS = 50
	LEARNING_RATE = 1e-3
	latent_space_dms = [3,6]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	for latent_space_dm in latent_space_dms:
		print("\n\n\nNEW RUN: VAE ",latent_space_dm)
		logfilename=f"../logs/CIFAR10_GAN_{latent_space_dm}.log"
		f = open(logfilename, "w")
		f.close()
		logging.basicConfig(filename=logfilename,level=logging.INFO)
		main(latent_space_dm)
