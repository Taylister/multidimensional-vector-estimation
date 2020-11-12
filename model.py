import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from layers import Flatten, Concatenate

class VGG(nn.Module):
    def __init__(self,input_img_shape, n_class, n_dimention=1):
        super(VGG, self).__init__()

        self.input_img_shape = input_img_shape
        self.n_class = n_class
        self.output_shape = (n_dimention,)
        self.img_c, self.img_h, self.img_w = input_img_shape 

        # the first argument of nn.Conv2d is the number of the channel which enter the layer
        # input_shape: (None, img_c + n_class, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c+self.n_class, 64, 3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.act3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.act4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.act4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 512, img_h//16, img_w//16)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.act5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.act5_2 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = Flatten()

        self.linear6 = nn.Linear(in_features, 4096)
        self.act6 = nn.ReLU()
        self.drop6 = nn.Dropout2d()

        self.linear7 = nn.Linear(4096, 4096)
        self.act7 = nn.ReLU()
        self.drop7 = nn.Dropout2d()

        self.linear8 = nn.Linear(4096, n_dimention)
        self.act8 = nn.Identity()

    def forward(self, x):
    
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))


        x = self.act3_1(self.conv3_1(x))
        x = self.act3_2(self.conv3_2(x))
        x = self.pool3(x)


        x = self.act4_1(self.conv4_1(x))
        x = self.act4_2(self.conv4_2(x))
        x = self.pool4(x)


        x = self.act5_1(self.conv5_1(x))
        x = self.act5_2(self.conv5_2(x))
        x = self.pool5(x)

        x = self.act6(self.linear6(self.flatten6(x)))
        x = self.drop6(x)

        x = self.act7(self.linear7(x))
        x = self.drop7(x)

        x = self.act8(self.linear8(x))
        
        return x

class VGG_c(nn.Module):
    def __init__(self,input_img_shape, n_class, n_dimention=1):
        super(VGG_c, self).__init__()

        self.input_img_shape = input_img_shape
        self.n_class = n_class
        self.output_shape = (n_dimention,)
        self.img_c, self.img_h, self.img_w = input_img_shape 

        # the first argument of nn.Conv2d is the number of the channel which enter the layer
        # input_shape: (None, img_c + n_class, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c, 64, 3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.act3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.act4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.act4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 512, img_h//16, img_w//16)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.act5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.act5_2 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

         # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = Flatten()

        self.linear6 = nn.Linear(in_features, 4096)
        self.act6 = nn.ReLU()
        self.drop6 = nn.Dropout2d()

        self.linear7 = nn.Linear(4096, 4096)
        self.act7 = nn.ReLU()
        self.drop7 = nn.Dropout2d()

        self.linear8 = nn.Linear(4096, n_class)

    def forward(self, x):
    
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))


        x = self.act3_1(self.conv3_1(x))
        x = self.act3_2(self.conv3_2(x))
        x = self.pool3(x)


        x = self.act4_1(self.conv4_1(x))
        x = self.act4_2(self.conv4_2(x))
        x = self.pool4(x)


        x = self.act5_1(self.conv5_1(x))
        x = self.act5_2(self.conv5_2(x))
        x = self.pool5(x)

        x = self.act6(self.linear6(self.flatten6(x)))
        x = self.drop6(x)

        x = self.act7(self.linear7(x))
        x = self.drop7(x)

        x = self.linear8(x)
        
        return x
class Network(torch.nn.Module):
    def __init__(self,input_img_shape, n_class, n_dimention=1):
        super(Network, self).__init__()
        self.input_img_shape = input_img_shape
        self.n_class = n_class
        self.output_shape = (n_dimention,)
        self.img_c, self.img_h, self.img_w = input_img_shape 

        # the first argument of nn.Conv2d is the number of the channel which enter the layer
        # input_shape: (None, img_c + n_class, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c+self.n_class, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()

        # input_shape: (None, 512, img_h//16, img_w//16)
        # output_shape: (None, 512, img_h//32, img_w//32)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()

        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = Flatten()

        # torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()

        self.linear7 = nn.Linear(1024,n_dimention)
        self.act7 = nn.Identity()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        x = self.act7(self.linear7(x))
        return x
