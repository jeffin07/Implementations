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



if __name__ == '__main__':
	
	inp = torch.randn(1,3,224,224)
	pat = Patches(3, 32, 512, 224)
	print(pat(inp).shape)
