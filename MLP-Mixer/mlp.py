import torch
import torch.nn as nn
import einops


class Patches(nn.Module):
	"""Patches Class
	
	Arguments
	---------

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

	"""MLP Block Class

	Arguments
	---------

	patch_dim : int 
				patch dimension
	hidden_dim : int
				projection dimension between layers
	

	"""

	def __init__(self, patch_dim, hidden_dim):

		super(MLP_Block, self).__init__()

		self.patch_dim = patch_dim
		self.hidden_dim = hidden_dim

		self.layer1 = nn.Linear(self.patch_dim, self.hidden_dim)
		self.layer2 = nn.Linear(self.hidden_dim, self.patch_dim)
		self.gelu = nn.GELU()


	def forward(self, x):

		x = self.layer1(x)
		x = self.gelu(x)
		x = self.layer2(x)	 
		return x


class Mixer_Block(nn.Module):

	
	"""Mixer_Block Class

	Arguments
	---------

		n_patches : int
					patch dimension
		hidden_dim : int 
					input dimension for token mlp
		channel_dim : int 
					output dimension for channel mlp
		token_dim : int
					output dimension for token mlp


	"""


	def __init__(self, n_patches, hidden_dim, channel_dim, token_dim,):

		super(Mixer_Block, self).__init__()

		self.n_patches = n_patches
		self.hidden_dim = hidden_dim
		self.channel_dim = channel_dim
		self.token_dim = token_dim


		self.norm1 = nn.LayerNorm(hidden_dim)
		self.norm2 = nn.LayerNorm(hidden_dim)
		self.token_mlp = MLP_Block(self.n_patches, self.channel_dim)
		self.channel_mlp = MLP_Block(self.hidden_dim, self.token_dim)

	
	def forward(self, x):

		out = self.norm1(x)
		out = out.transpose(1,2)
		out = self.token_mlp(out)
		out = out.transpose(1,2)
		y = out + x
		out = self.norm2(out)
		out = self.channel_mlp(out)
		out = out + y

		return out


class MLP_Mixer(nn.Module):

	"""MLP_Mixer Class

	Arguments
	---------
		img_dim : int
					input image dimension
		patch_size : int
					patch size dimension
		hidden_dim : int
					dimension for the projections
		channel_dim : int
					channel_mlp hidden dimension
		token_dim : int
					token_mlp hidden_dimension
		n_block : int
					number of mlp
		n_classes : int
					number of classes

	"""

	def __init__(self, img_dim, patch_size, hidden_dim, channel_dim, token_dim, n_blocks, n_classes):

		super(MLP_Mixer,self).__init__()

		self.img_dim = img_dim
		self.patch_size = patch_size
		self.hidden_dim = hidden_dim
		self.channel_dim = channel_dim
		self.token_dim = token_dim
		self.n_blocks = n_blocks
		self.n_classes = n_classes

		self.n_patches = (self.img_dim // self.patch_size ) ** 2
		self.mixer_blocks = nn.ModuleList(
			[
				Mixer_Block(self.n_patches, self.hidden_dim, self.channel_dim, self.token_dim)
				for _ in range(self.n_blocks)
			]
			
		)

		self.norm1 = nn.LayerNorm(hidden_dim)
		self.classifier = nn.Linear(hidden_dim, self.n_classes)

	def forward(self, x):

		
		x = einops.rearrange(
            x, "n c h w -> n (h w) c")
		for mixer_block in self.mixer_blocks:

			out = mixer_block(x)
		
		out = self.norm1(out)
		out = out.mean(dim=1)
		pred = self.classifier(out)

		return pred
