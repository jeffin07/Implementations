import torch
import torch.nn as nn


class Patches(nn.Module):
	"""Patches Class
	
	Paramter
	--------

	in_channels : int
				Input channel number 
	patch_size : int
				Patch size
	hidden_dim : int
				output channels for the patch
	img_size :  int
				Input image dimension


	"""
	def __init__(self, in_channels, patch_size, hidden_dim, img_size):
		
		super(Patches, self).__init__()
		
		self._cls = 'Patches'
		self.in_channels = in_channels
		self.patch_size = patch_size
		self.img_size = img_size
		self.hidden_dim = hidden_dim
		self.num_patches = (self.img_size // self.patch_size) ** 2

		self.patches = nn.Conv2d(
			in_channels = self.in_channels, out_channels = self.hidden_dim,
			kernel_size = self.patch_size, stride = self.patch_size)

	def forward(self, x):

		return self.patches(x)

class MLP_Block(nn.Module):

	"""
	
	simple class two linear layer with gelu

	"""

	def __init__(self, patch_dim, hidden_dim):

		super(MLP_Block, self).__init__()

		self.patch_dim = patch_dim
		self.hidden_dim = hidden_dim

		self.layer1 = nn.Linear(self.patch_dim, self.hidden_dim)
		self.layer2 = nn.Linear(self.hidden_dim, self.patch_dim)
		self.gelu = nn.GELU()


	def forward(self, x):

		x = self.layer2(self.gelu(self.layer1(x)))
		return x


class Mixer_Block(nn.Module):

	
	"""
		norm -> xT -> mlp1 ->xT ->norm ->mlp2.xT ->

	"""


	def __init__(self, n_patches, hidden_dim, channel_dim, token_dim,):

		super(Mixer_Block, self).__init__()

		self.n_patches = n_patches
		self.hidden_dim = hidden_dim
		self.channel_dim = channel_dim
		self.token_dim = token_dim


		self.norm1 = nn.LayerNorm(hidden_dim)
		self.norm2 = nn.LayerNorm(hidden_dim)
		self.channel_mlp = MLP_Block(self.hidden_dim, self.channel_dim)
		self.token_mlp = MLP_Block(self.hidden_dim, self.token_dim)
		# [mlp1]
		# [mlp2]
	
	def forward(self, x):

		out = self.norm1(x)
		out = out.transpose(1,2)
		out = self.channel_mlp(out)
		out = out.transpose(1,2)
		y = out + x
		out = self.norm2(out)
		out = self.channel_mlp(out)
		out = out + y

		return out


if __name__ == '__main__':
	
	inp = torch.randn(1,3,224,224)
	pat = Patches(3, 32, 512, 224)
	print(pat(inp).shape)
