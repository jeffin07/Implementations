import torch
import torch.nn as nn
from torch.nn import Sequential, ModuleList

class ConvStage(nn.Module):
	
	def __init__(self, num_blocks=2,input_dim=64,  hidden_dim=128, out_dim=128):
		
		super(ConvStage, self).__init__()
		
		self.num_blocks = num_blocks
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.out_dim  = out_dim
		#self.batchnorm = nn.BatchNorm2d(self.hidden_dim)
		self.relu = nn.ReLU()
		self.blocks = ModuleList()
		for i in range(self.num_blocks):
			block = Sequential(
					nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=(1,1), padding=(0,0)),
					nn.BatchNorm2d(self.hidden_dim),
					self.relu,
					nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3,3),padding=(1,1)),
					nn.BatchNorm2d(self.hidden_dim),
					self.relu,
					nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size=(1,1), padding=(0,0)),
					nn.BatchNorm2d(self.input_dim),
					self.relu
				)
			self.blocks.append(block)
		self.downsample = nn.Conv2d(self.input_dim, self.out_dim,kernel_size=(3,3), stride=(2,2),padding=(1,1))


	def forward(self,x):

		for block in self.blocks:
			x = x + block(x)
		return self.downsample(x)


if __name__ == '__main__':

	test = torch.randn((1,64,56,56))
	c = ConvStage()
	print(c(test).shape)
