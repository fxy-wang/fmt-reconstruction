"""
Neural Network for FMT Reconstruction
- Model Definitions for UNet and AE
- Only UNet is used in main, scripts can be tweaked to use only AE or AE + UNet combo

Created by Fay Wang
Contact: fay.wang@columbia.edu

"""

# Import packages
from ops import *
import torch
import torch.nn as nn
import torch.nn.functional as F

#============#
# UNET MODEL #
#============#
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 n_blocks: int = 4,
                 start_filters: int = 32,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode: str = 'transposed',
                 input_size = 768,
                 output_size=11094
                 ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode
        
        self.down_blocks = []
        self.up_blocks = []
        
        self.l1 = nn.Linear(input_size,output_size)
        self.l2= nn.Linear(output_size,output_size)
        
        # create the encoder pathway
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False
            
            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode,
                                   dim=self.dim)
            
            self.down_blocks.append(down_block)
        
        # create the decoder pathway (only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2
            
            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               conv_mode=self.conv_mode,
                               dim=self.dim,
                               up_mode=self.up_mode)
            
            self.up_blocks.append(up_block)
        
        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True, dim=self.dim)
        
        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        
        # initialize the weights
        self.initialize_parameters()
    
    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights
    
    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias
    
    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias
    
    def forward(self, x: torch.tensor):
        # forward pass for the model
        encoder_output = [] # only used if being used w/ AE
        x = self.l1(x) # maps from measurement space to parameter space
        x = self.l2(x) 
        x = reshape_fortran(x,[x.size(0),41,41,5]) # reshape to 3D medium
        x = torch.unsqueeze(x,1)
        x = F.pad(x,(0,3,0,3,0,3)) # pad to appropriate size 
        
        # encoder path
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)
        
        # decoder path
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)
        
        # final convolution
        x = self.conv_final(x)
        
        return x
    
    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
    

#===================#
# Convolutional AE  #
#===================#
class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, medium_size, code_size):
        super(ConvAutoencoder, self).__init__()
        self.code_size = code_size
        # linear layers to map from measurement to parameter space
        self.l1 = nn.Linear(input_size,medium_size)
        self.l2 = nn.Linear(medium_size, medium_size)
        # encoder
        self.e_conv1 = nn.Conv3d(1, 8, 3, padding=1)  
        self.e_conv2 = nn.Conv3d(8, 4, 3, padding=1)
       
        # decoder
        self.t_conv1 = nn.ConvTranspose3d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose3d(16, 1, 2, stride=2)

    def encode(self, images):
        # encoder pathway 
        code = self.e_conv1(images)
        code = F.selu(F.max_pool3d(code, 2))
        
        code = self.e_conv2(code)
        code = F.selu(F.max_pool3d(code, 2))
        
        return code
    
    def decode(self, code):
        # decoder pathway
        out = F.relu(self.t_conv1(code))
        out = F.relu(self.t_conv2(out))
        return out
    
    def forward(self, meas):
        # forward pass definition 
        images = self.l1(meas) # maps from measurement space to parameter space
        images = self.l2(images)
        images=reshape_fortran(images,[meas.size(0),41,41,5]) # reshape to 3D medium
        images = torch.unsqueeze(images,1)
        code = self.encode(images) # encode 
        out = self.decode(code) # decode
        out = F.interpolate(out, size=[41,41,5], mode='nearest') # interpolate to medium size
        # out = F.pad(out,(0,0,0,3,0,3)) # pad if being input into UNet
        return out, code