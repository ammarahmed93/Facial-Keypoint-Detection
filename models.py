## TODO: define the convolutional neural network architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (16, 222, 222)
        # after one pool layer, the output Tensor for one image will become (16, 111, 111)
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv1_bn = nn.BatchNorm2d(16)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define a second conv layer
        # 32 input image channel (grayscale), 64 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (111-3)/1 + 1 = 109
        # the output Tensor for one image, will have the dimensions: (32, 109, 109)
        # after one pool layer, the output Tensor for one image will become (32, 54, 54) $ not used 
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        # Define a 3rd conv layer
        # 64 input image channel (grayscale), 128 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for one image, will have the dimensions: (64, 52, 52)
        # after one pool layer, the output Tensor for one image will become (64, 26, 26)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        # Define a 4th conv layer
        # 64 input image channel (grayscale), 256 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output Tensor for one image, will have the dimensions: (128, 24, 24)
        # after one pool layer, the output Tensor for one image will become (128, 12, 12)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv4_bn = nn.BatchNorm2d(128)
        
        # Define a 5th conv layer
        # 64 input image channel (grayscale), 256 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output Tensor for one image, will have the dimensions: (256, 10, 10)
        # after one pool layer, the output Tensor for one image will become (256, 5, 5)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv5_bn = nn.BatchNorm2d(256)
        
        # fully-connected layer
        self.fc1 = nn.Linear(256*5*5, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        #dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels (for the 136 classes)
        self.fc2 = nn.Linear(512, 136)
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # one linear layer
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc1_drop(x)
        x = self.fc2(x)
    
        # a modified x, having gone through all the layers of the model, should be returned
        return x
    
    
