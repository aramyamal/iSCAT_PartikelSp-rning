import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First block of three 3x3 conv layers with 32 filters each
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block of eight 3x3 conv layers with 32 filters each
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        # Final 1x1 conv layer with 3 filters
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        
    def forward(self, x):
        # Apply the first three convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply max pooling
        x = self.pool(x)
        
        # Apply the next eight convolutional layers with ReLU
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        
        # Apply the final convolutional layer without activation
        x = self.final_conv(x)
        
        return x
    
    def __call__(self,x):
        return self.forward(x)