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


class DWBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.dw1 = DWUnit(self.input_channels, self.input_channels)
        self.dw2 = DWUnit(self.input_channels, self.output_channels)
    
    def forward(self, x):

        out1 = self.dw1(x)
        out2 = self.dw2(out1)

        return out2

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
        self.dwunit = DWUnit(self.output_channels, self.output_channels)
    
    def forward(self, x):

        conv1 = self.conv2d(x)
        bn = self.batchnorm(conv1)
        act = self.relu(bn)
        out = self.dwunit(act)
        return out




if __name__ == "__main__":

    input = torch.randn(1, 3, 640, 640)
    # stage 0
    convhead = convHead()
    convhead_out = convhead(input)
    print(convhead_out.shape)
    # stage 1
    dwblock1 = DWBlock(input_channels=16, output_channels=16)
    dwblock2 = DWBlock(input_channels=16, output_channels=64)
    
    maxpool = nn.MaxPool2d(2)
    m_out = maxpool(convhead_out)
    dwblock1_out = dwblock1(m_out)
    dwblock2_out = dwblock2(dwblock1_out)
    print(dwblock2_out.shape)
    # stage 2
    dwblock3 = DWBlock(input_channels=64, output_channels=64)
    dwblock3_out = dwblock3(dwblock2_out)
    m_out1 = maxpool(dwblock3_out)
    print(m_out1.shape)
    # stage 3
    dwblock4 = DWBlock(input_channels=64, output_channels=64)
    dwblock4_out = dwblock3(m_out1)
    m_out2 = maxpool(dwblock4_out)
    print(m_out2.shape)
    # stage 4
    dwblock5 = DWBlock(input_channels=64, output_channels=64)
    dwblock5_out = dwblock3(m_out2)
    m_out3 = maxpool(dwblock5_out)
    print(m_out3.shape)