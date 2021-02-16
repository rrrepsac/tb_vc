# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:30:55 2021

@author: Kobap
"""

import torch
from torch import nn
from torchvision import transforms
from torch.nn import ModuleList, InstanceNorm2d
import numpy as np


#from matplotlib import pyplot as plt
from pathlib import Path
from torch.nn import Sequential, ReLU, Conv2d, AvgPool2d, ReflectionPad2d,\
BatchNorm2d, InstanceNorm2d, UpsamplingNearest2d
#from torch.nn import ConvTranspose2d as ConvT2d
#from functools import lru_cache

from PIL import Image
import time

transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)


#DEVICE = (torch.device("cpu"), torch.device("cuda"))[torch.cuda.is_available()]
DEVICE = torch.device('cpu')

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
#inv_mean, inv_std = zip(*[(-m/s, 1/s)for (m, s) in zip(mean, std)])

transform_norm = transforms.Normalize(mean, std, False)
#transform_denorm=transforms.Normalize(inv_mean, inv_std, False)

transform_inference = transforms.Compose([transforms.ToTensor(), transform_norm])


class Residual(nn.Module):
  def __init__(self, features_number, kernel_size=3):
    super().__init__()
    self.conv2d_1 = Conv2d(features_number, features_number, kernel_size, 1, padding=1, padding_mode='reflect')
    self.act   = ReLU()
    self.norm_1   = InstanceNorm2d(features_number)
    self.conv2d_2 = Conv2d(features_number, features_number, kernel_size, 1, padding=1, padding_mode='reflect')
    self.norm_2   = InstanceNorm2d(features_number)
  def forward(self, x):
    return x + \
    self.norm_2(self.conv2d_2(
    self.act(self.norm_1(self.conv2d_1(x)))))

class BottleneckResidual(nn.Module):
  def __init__(self, features_number, kernel_size=3):
    super().__init__()
    self.conv2d_1 = Conv2d(features_number, features_number//8, 1, 1, padding=0)
    self.act   = ReLU()
    self.norm_1   = InstanceNorm2d(features_number//8)
    self.conv2d_2 = Conv2d(features_number//8, features_number//8, kernel_size, 1, padding=1, padding_mode='reflect')
    self.norm_2   = InstanceNorm2d(features_number)
    self.conv2d_3 = Conv2d(features_number//8, features_number, 1, 1, padding=0)
    self.norm_3   = InstanceNorm2d(features_number)
  def forward(self, x):
    return x + \
            self.norm_3(self.conv2d_3(
    self.act(self.norm_2(self.conv2d_2(
    self.act(self.norm_1(self.conv2d_1(x)))
    )))))
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(
                mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
            
class Johnson_net(nn.Module):
  def __init__(self, mode='bottleneck'):
    super().__init__()
    features_list=[32,64,128]
    norm = InstanceNorm2d
    if mode == 'bottleneck':
      res = BottleneckResidual
    else:
      res = Residual
    self.downsample = Sequential(
                                 Conv2d(3, features_list[0], kernel_size=9, stride=1, padding=4, padding_mode='reflect'), norm(features_list[0]), ReLU(),
                                 Conv2d(features_list[0], features_list[1], 3, 2, 1, padding_mode='reflect'), norm(features_list[1]), ReLU(),
                                 Conv2d(features_list[1], features_list[2], 3, 2, 1, padding_mode='reflect'), norm(features_list[2]), ReLU(),
                                 )
    self.res = Sequential(res(features_list[-1]), res(features_list[-1]), res(features_list[-1]), 
                          res(features_list[-1]), res(features_list[-1])
                          )
    self.upsample = Sequential(UpsamplingNearest2d(2),
                               Conv2d(features_list[2], features_list[1], 3, 2, 1, padding_mode='reflect'), norm(features_list[2]), ReLU(),
                               UpsamplingNearest2d(2),
                               Conv2d(features_list[1], features_list[0], 3, 2, 1, padding_mode='reflect'), norm(features_list[1]), ReLU(),
                               Conv2d(features_list[0], 3, kernel_size=9, stride=1, padding=4, padding_mode='reflect'), norm(features_list[0])#, ReLU(),
                                 
                               
                               
                               #ConvT2d(features_list[-1], features_list[-2], 3, stride=2, padding=0), norm(features_list[-2]), ReLU(),
                               #ConvT2d(features_list[-2], features_list[-3], 3, stride=2, padding=0), norm(features_list[-3]), ReLU(),
                               #Conv2d(features_list[-3], 3, 8, stride=1, padding=2, padding_mode='reflect'), norm(3)#, ReLU(),
                               #Conv2d(3, 3, 2, stride=1, padding=0), norm(3)
                                
                               )
    self.act = nn.Sigmoid() #nn.Tanh()
  def forward(self, x):
    x = self.downsample(x)
   # print(f'after dsmpl {x.shape}')
    x = self.res(x)
   # print(f'after res {x.shape}')
    x = self.upsample(x)
   # print(f'after up {x.shape}')

    return self.act(x)
    
class Pyramid(nn.Module):
  def __init__(self, mode=None):
    super().__init__()
    ratios = [32, 16, 8, 4, 2, 1]
    mode = 5
    if mode == 5:
        ratios = ratios[1:]
    self.stages = len(ratios)
    self.mode = mode
    do_inplace = False  # True
    self.downs = nn.ModuleList([ReLU()]*self.stages) #
    self.ups   = nn.ModuleList([ReLU()]*self.stages) #
    norm = InstanceNorm2d
    features = 8*1
    act = ReLU#nn.LeakyReLU #ReLU
    class down_com(nn.Module):
      def __init__(self, layer_number):
        super().__init__()
        self.layer = Sequential(
          AvgPool2d(ratios[layer_number], ratios[layer_number]),
          Conv2d(3, features, 3, padding=1, padding_mode='reflect'), norm(features), act(do_inplace),
          Conv2d(features, features, 3, padding=1, padding_mode='reflect'), norm(features), act(do_inplace),
          Conv2d(features, features, 1), norm(features), act(do_inplace),
          #norm(features)
        )
      
    #self.downsdown_com(0).layer)
     #self.downs[0] = down_com(0).layer  # Sequential(#norm(3),
        #*down_com(0).layer[:-1], UpsamplingNearest2d(scale_factor=2), norm(features))
    for layer_number in range(0, self.stages):
      self.downs[layer_number] = down_com(layer_number).layer
      #self.downs.append(down_com(layer_number).layer)
    
    class up_com(nn.Module):
      def __init__(self, features_num):
        super().__init__()
        self.layer = Sequential(
                norm(features_num),
                Conv2d(features_num, features_num, 3, padding=1, padding_mode='reflect'), norm(features_num), act(do_inplace),
                Conv2d(features_num, features_num, 3, padding=1, padding_mode='reflect'), norm(features_num), act(do_inplace),
                Conv2d(features_num, features_num, 1), norm(features_num), act(do_inplace),
                #UpsamplingNearest2d(scale_factor=2),
                
                #Conv2d(features_num, features_num, 3, padding=1, padding_mode='reflect'), norm(features_num), act(do_inplace),
                UpsamplingNearest2d(scale_factor=2),
                #Conv2d(features_num, features_num, 3, padding=1, padding_mode='reflect'), norm(features_num), act(do_inplace),
                )
        
    for layer_number in range(self.stages):
      features_num = (layer_number + 1)*features
      self.ups[layer_number] = up_com(features_num).layer
      #self.ups.append(up_com(features_num).layer)
    self.ups[layer_number] = Sequential(*self.ups[layer_number][:-1], Conv2d(features_num, 3, 1))
    
  def forward(self, image):
    last_out = None
    #image = self.initial_bn(image.clone())
    for stage in range(self.stages):
      #print(f'stage={stage}, {image.device}')
      
      cur_x = self.downs[stage](image.clone())
      #print('inp = ', image.shape, f'down({stage})=', cur_x.shape)
      tocat = [cur_x]
      if last_out is not None:
        tocat.append(last_out)
      #print(f'{[x.shape for x in tocat]}')
      cur_x = torch.cat(tocat, dim=1)
      last_out = self.ups[stage](cur_x) # del clone
      #print(cur_x.shape, last_out.shape)
    return last_out


class JohnsonMultiStyleNet(torch.nn.Module):
    def get_style_number(self):
        return self.style_number
        
    def __init__(self, style_number = 1):
        super().__init__()
        self.style_number = style_number
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = ModuleList([InstanceNorm2d(32, affine=True) for _ in range(style_number)])        
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = ModuleList([InstanceNorm2d(64, affine=True) for _ in range(style_number)])  
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = ModuleList([InstanceNorm2d(128, affine=True) for _ in range(style_number)])  
        # Residual layers
        self.res1 = ResidualMultiStyleBlock(128, style_number)
        self.res2 = ResidualMultiStyleBlock(128, style_number)
        self.res3 = ResidualMultiStyleBlock(128, style_number)
        self.res4 = ResidualMultiStyleBlock(128, style_number)
        self.res5 = ResidualMultiStyleBlock(128, style_number)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(
            128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = ModuleList([InstanceNorm2d(64, affine=True) for _ in range(style_number)])  
        self.deconv2 = UpsampleConvLayer(
            64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = ModuleList([InstanceNorm2d(32, affine=True) for _ in range(style_number)])  
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        model_pth = f'multistyle256_{style_number}.pth'
        if Path(model_pth).exists():
            print('loading model...')
            self.load_state_dict(torch.load(model_pth, map_location=DEVICE))

    def forward(self, X, style):
        y = self.relu(self.in1[style](self.conv1(X)))
        y = self.relu(self.in2[style](self.conv2(y)))
        y = self.relu(self.in3[style](self.conv3(y)))
        y = self.res1(y, style)
        y = self.res2(y, style)
        y = self.res3(y, style)
        y = self.res4(y, style)
        y = self.res5(y, style)
        y = self.relu(self.in4[style](self.deconv1(y)))
        y = self.relu(self.in5[style](self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ResidualMultiStyleBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, style_number=1):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = ModuleList([InstanceNorm2d(channels, affine=True) for _ in range(style_number)])  
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = ModuleList([InstanceNorm2d(channels, affine=True) for _ in range(style_number)])  
        self.relu = torch.nn.ReLU()

    def forward(self, x, style):
        residual = x
        out = self.relu(self.in1[style](self.conv1(x)))
        out = self.in2[style](self.conv2(out))
        out = out + residual
        return out
def make_style(img, style_model, style_choice=None):
    img_t = transform_inference(img).unsqueeze(0)

    style_num = style_model.get_style_number()
    if style_choice is None:
        style_choice = np.random.randint(style_num)
    style_model.eval()
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        styled = style_model(img_t, style_choice)
    return recover_image(styled.detach().cpu().numpy())[0]
