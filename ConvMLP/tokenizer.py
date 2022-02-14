import torch
import torch.nn as nn


class Tokenizer(nn.Module):
	
	def __init__(self, embedding_dim=64):
		
		super(Tokenizer, self).__init__()
		self.embedding_dim = embedding_dim
		# need to know more !
		self.tdim = self.embedding_dim // 2
		self.tkernel = (3, 3)
		self.tconv1 = nn.Conv2d(3, self.tdim, self.tkernel, (2, 2))
		self.tconv2 = nn.Conv2d(self.tdim, self.tdim, self.tkernel, (1, 1))
		self.tconv3 = nn.Conv2d(self.tdim, self.embedding_dim, self.tkernel, (1, 1))
		self.activation = nn.ReLU(inplace = True)
		self.batchnorm1 = nn.BatchNorm2d(self.tdim)
		self.batchnorm2 = nn.BatchNorm2d(self.embedding_dim)
		self.maxpool = nn.MaxPool2d(3, stride = 2)

	def forward(self, x):
		
		x = self.activation(self.batchnorm1(self.tconv1(x)))
		x = self.activation(self.batchnorm1(self.tconv2(x)))
		x = self.activation(self.batchnorm2(self.tconv3(x)))
		x = self.maxpool(x)
		
		return x




if __name__ == '__main__':
	
	a = torch.randn((1, 3,224,224))
	tok = Tokenizer()
	res = tok(a)
	print(res.shape)
