import torch
import torch.nn as nn



class DWUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # depthiwse
        self.depth = nn.Conv2d(
            in_channels=self.in_channels, out_channels = self.in_channels,
            kernel_size=self.kernel_size, groups=self.in_channels)
        # pointwise
        self.pointwise = nn.Conv2d(
            in_channels=self.in_channels, out_channels = self.out_channels,
            kernel_size=1, padding=self.padding)
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        depth = self.depth(x)
        pointwise = self.pointwise(depth)
        bn = self.batchnorm(pointwise)
        act = self.relu(bn)
        
        return act

        

class convHead(nn.Module):
    def __init__(self, input_channels=3, output_channels=16, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv2d = nn.Conv2d(
            in_channels=self.input_channels, out_channels=self.output_channels,
            kernel_size=self.kernel_size, stride=self.stride, padding = self.padding)
        self.batchnorm = nn.BatchNorm2d(self.output_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        conv1 = self.conv2d(x)
        bn = self.batchnorm(conv1)
        act = self.relu(bn)
        return act




if __name__ == "__main__":

    input = torch.randn(1, 3, 100, 100)
    convhead = convHead()
    dwunit = DWUnit(in_channels=16, out_channels=64)
    convhead_out = convhead(input)
    print(convhead_out.shape)
    maxpool = nn.MaxPool2d(2)
    m_out = maxpool(convhead_out)
    dw_out = dwunit(m_out)
    print(dw_out.shape)