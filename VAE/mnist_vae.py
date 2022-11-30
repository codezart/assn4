import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image,  make_grid
import matplotlib
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

class VAE(nn.Module):
	def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
		super(VAE, self).__init__()

		#Encoding layers
		self.fc1 = nn.Linear(x_dim, h_dim1)
		self.fc2 = nn.Linear(h_dim1, h_dim2)
		self.fc3_mu = nn.Linear(h_dim2, z_dim)
		self.fc3_log_var = nn.Linear(h_dim2, z_dim)

		#Decoding layers
		self.fc4 = nn.Linear(z_dim, h_dim2)
		self.fc5 = nn.Linear(h_dim2, h_dim1)
		self.fc6 = nn.Linear(h_dim1, x_dim)

	def encoder(self, x):
		h = F.relu(self.fc1(x))
		h = F.relu(self.fc2(h))
		mu = self.fc3_mu(h)
		log_var = self.fc3_log_var(h)
		return mu, log_var

	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5*log_var) 
		eps = torch.randn_like(std)
		sample = mu + eps * log_var
		return sample

	def decoder(self, z):
		h = F.relu(self.fc4(z))
		h = F.relu(self.fc5(h))
		reconstruction = torch.sigmoid(self.fc6(h))
		return reconstruction

	def forward(self, x):
		mu, log_var = self.encoder(x.view(-1, 784))
		z = self.reparameterize(mu, log_var)
		return self.decoder(z), mu, log_var 



def main(x_dim):

	transform = transforms.Compose([
						transforms.ToTensor(),
	])

	train_dataset = torchvision.datasets.MNIST(root='../DATA_MNIST',train=True, transform=transform, download=True)
	test_dataset = torchvision.datasets.MNIST(root='../DATA_MNIST',train=False, transform=transform, download = True)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


	model = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2).to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_losses = []
	train_losses_BCE = []
	train_losses_KLD = []

	test_losses = []
	test_losses_BCE = []
	test_losses_KLD = []

	epoch_list = []
	for epoch in range(EPOCHS):
			print(f"Epoch {epoch+1} of {EPOCHS}")
			train_epoch_loss_BCE, train_epoch_loss_KLD, train_epoch_loss = train(model, optimizer, train_loader)
			test_epoch_loss_BCE, test_epoch_loss_KLD, test_epoch_loss = evaluate(model, optimizer, test_loader, BATCH_SIZE, epoch,x_dim)
			train_losses.append(train_epoch_loss)
			train_losses_BCE.append(train_epoch_loss_BCE)
			train_losses_KLD.append(train_epoch_loss_KLD)
			test_losses.append(test_epoch_loss)
			test_losses_BCE.append(test_epoch_loss_BCE)
			test_losses_KLD.append(test_epoch_loss_KLD)
			epoch_list.append(epoch+1)
			print(f"Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}, Test BCE Loss: {test_epoch_loss_BCE:.4f}, Test KLD Loss: {train_epoch_loss_KLD:.4f}")

	fig = plt.figure()
	plt.plot(epoch_list, train_losses, color='black', label = "train loss")
	plt.plot(epoch_list, test_losses, color='red', label = "test loss")
	plt.legend(loc='best', prop={'size': 10})
	plt.xlabel('Epoch')
	plt.ylabel('Reconstruction Loss and KL Divergence')
	plt.title('VAE_MNIST_Reconstruction_Loss_and_KL_Divergence')
	plt.savefig(f"plots/MNIST/{x_dim}_vae_total_loss.png")
	plt.clf()
	plt.close()


	fig = plt.figure()
	plt.plot(epoch_list, train_losses_BCE, color='black', label = "train loss")
	plt.plot(epoch_list, test_losses_BCE, color='red', label = "test loss")
	plt.legend(loc='best', prop={'size': 10})
	plt.xlabel('Epoch')
	plt.ylabel('Binary Cross Entropy')
	plt.title('VAE_MNIST_Reconstruction_Loss')
	plt.savefig(f"plots/MNIST/{x_dim}_vae_BCE.png")
	plt.clf()
	plt.close()



	fig = plt.figure()
	plt.plot(epoch_list, train_losses_KLD, color='black', label = "train loss")
	plt.plot(epoch_list, test_losses_KLD, color='red', label = "test loss")
	plt.legend(loc='best', prop={'size': 10})
	plt.xlabel('Epoch')
	plt.ylabel(' KL Divergence')
	plt.title('VAE_MNIST_KL_Divergence')
	plt.savefig(f"plots/MNIST/{x_dim}_vae_KLD.png")
	plt.clf()
	plt.close()




def loss_function(recon_x, x, mu, log_var):
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction = 'sum')
   
	KLD = -0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())

	return BCE, KLD, BCE + KLD


def train(model, optimizer, train_loader):
	model.train()
	running_loss = 0.0
	running_loss_BCE = 0.0
	running_loss_KLD = 0.0
	for batch_idx, (images, _ ) in tqdm(enumerate(train_loader), total = int(len(train_loader.dataset) / BATCH_SIZE)):
			images = images.to(device)
			images = images.reshape(-1, 784)
			optimizer.zero_grad()
			recon_batch, mu, log_var = model(images)
			BCE_loss, KLD_loss, final_loss = loss_function(recon_batch, images, mu, log_var)
			final_loss.backward()
			running_loss += final_loss.item()
			running_loss_BCE += BCE_loss.item()
			running_loss_KLD += KLD_loss.item()
			optimizer.step()
	train_loss = running_loss/len(train_loader.dataset)
	train_loss_BCE = running_loss_BCE/len(train_loader.dataset)
	train_loss_KLD = running_loss_KLD/len(train_loader.dataset)
	return train_loss_BCE, train_loss_KLD, train_loss


def evaluate(model, optimizer, test_loader, batch_size, epoch,x_dim):
	model.eval()
	running_loss = 0.0
	running_loss_BCE = 0.0
	running_loss_KLD = 0.0
	with torch.no_grad():
			for batch_idx, (images, _ ) in enumerate(test_loader):
					images = images.to(device)
					images = images.reshape(-1, 784)
					recon_batch, mu, log_var = model(images)
					BCE_loss, KLD_loss, final_loss = loss_function(recon_batch, images, mu, log_var)
					running_loss += final_loss.item()
					running_loss_BCE += BCE_loss.item()
					running_loss_KLD += KLD_loss.item()

					if batch_idx == int(len(test_loader.dataset)/batch_size) - 1:
									recon_batch_ = recon_batch.view(batch_size, 1, 28, 28)[:64]
									generated_img = make_grid(recon_batch_, padding =2, normalize = True)
									save_generator_image(generated_img, f"plots/MNIST/output/model_{x_dim}_epoch{epoch+1}.png")

	test_loss = running_loss/len(test_loader.dataset)
	test_loss_BCE = running_loss_BCE/len(test_loader.dataset)
	test_loss_KLD = running_loss_KLD/len(test_loader.dataset)
	return test_loss_BCE, test_loss_KLD, test_loss


def save_generator_image(image, path):
	save_image(image, path)



if __name__=="__main__":
	  
	BATCH_SIZE = 128
	EPOCHS = 50
	LEARNING_RATE = 1e-3
	latent_space_dms = [64,784]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	for latent_space_dm in latent_space_dms:
		print("\n\n\nNEW RUN: VAE ",latent_space_dm)
		logfilename=f"../logs/CIFAR10_VAE_{latent_space_dm}.log"
		f = open(logfilename, "w")
		f.close()
		logging.basicConfig(filename=logfilename,level=logging.INFO)
		main(latent_space_dm)
