import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    # Base of the neural network introduced decades ago
    def __init__(self):
        super(Net,self).__init__() #(3, 32, 32)
        # 1: input channals 32: output channels, 3: kernel size, 1: stride
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        # Height & weidth of picture affected by (H&W-Kernel size)/stride+padding
        # in this case (32-5)/1+1 = 28

        self.pool = nn.AvgPool2d(2,2)  # max pool layer with 2x2 kernel size
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # picture shape （3,32,32） conv -> (32,28,28) maxpool -> (32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # picture shape （64,14,14）conv -> (64,10,10) maxpool -> (64,5,5)
        # x = self.pool(torch.sigmoid(self.conv1(x)))  # picture shape （3,32,32） conv -> (32,28,28) maxpool -> (32,14,14)
        # x = self.pool(torch.sigmoid(self.conv2(x)))  # picture shape （64,14,14）conv -> (64,10,10) maxpool -> (64,5,5)
        x = x.view(-1, 64 * 5 * 5) # flatten the size into one dimension
        # x = torch.sigmoid(self.fc1(x))   # use sigmoid activation function in respect to old standard
        # x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc1(x))   # use sigmoid activation function in respect to old standard
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
