import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def define_gridnet(input_nc, output_nc, ngf, use_dropout=False, gpu_ids=[]):
    gridnet = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    gridnet = GridNet_Network(input_nc, output_nc, ngf, use_dropout=use_dropout, gpu_ids=gpu_ids)
    
    if len(gpu_ids) > 0:
        gridnet.cuda(gpu_ids[0])
    gridnet.apply(weights_init)
    return gridnet

def define_pixelnet(input_nc, output_nc, ngf, use_dropout=False, gpu_ids=[]):
    pixelnet = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    pixelnet = PixelNet_Network(input_nc, output_nc, ngf, use_dropout=use_dropout, gpu_ids=gpu_ids)
    
    if len(gpu_ids) > 0:
        pixelnet.cuda(gpu_ids[0])
    pixelnet.apply(weights_init)
    return pixelnet

def define_depixelnet(input_nc, output_nc, ngf, use_dropout=False, gpu_ids=[]):
    depixelnet = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    depixelnet = DepixelNet_Network(input_nc, output_nc, ngf, use_dropout=use_dropout, gpu_ids=gpu_ids)
    
    if len(gpu_ids) > 0:
        depixelnet.cuda(gpu_ids[0])
    depixelnet.apply(weights_init)
    return depixelnet

def define_D(input_nc, ndf, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())
    netD = Discriminator(input_nc, ndf, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                real_tensor.requires_grad = False
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                fake_tensor.requires_grad = False
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class GridNet_Network(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, gpu_ids=[]):
        super(GridNet_Network, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        
        # model for downsampling
        model_0 = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=True),
                   norm_layer(ngf, track_running_stats=True),
                   nn.ReLU(True)]

        mult = 1
        model_1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=True),
                   norm_layer(ngf * mult * 2, track_running_stats=True),
                   nn.ReLU(True)]

        mult = 2
        model_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=True),
                   norm_layer(ngf * mult * 2, track_running_stats=True),
                   nn.ReLU(True)]
        
        mult = 4
        model_3 = []
        for i in range(9):
            model_3 += [ResnetBlock(ngf * mult, use_dropout=use_dropout)]
        
        model_4 = []
        for i in range(8):
            model_4 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                  stride=1, padding=0, bias=True),
                        norm_layer(ngf * mult, track_running_stats=True),
                        nn.ReLU(True)]
            
        model_5 = [] 
        for i in range(8):
            model_5 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                  stride=1, padding=0, bias=True),
                        norm_layer(ngf * mult, track_running_stats=True),
                        nn.ReLU(True)]
        
        model_output_x4 = [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf * mult, output_nc, kernel_size=7, padding=0),
                           nn.Tanh()]
        
        model_output_x6 = [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf * mult, output_nc, kernel_size=7, padding=0),
                           nn.Tanh()]
        
        model_output_x8 = [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf * mult, output_nc, kernel_size=7, padding=0),
                           nn.Tanh()]        

        self.model_0 = nn.Sequential(*model_0)
        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)
        self.model_4 = nn.Sequential(*model_4)
        self.model_5 = nn.Sequential(*model_5)
        self.model_output_x8 = nn.Sequential(*model_output_x8)
        self.model_output_x6 = nn.Sequential(*model_output_x6)        
        self.model_output_x4 = nn.Sequential(*model_output_x4)
        
        
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            feature_x1 = nn.parallel.data_parallel(self.model_0, input, self.gpu_ids)
            feature_x2 = nn.parallel.data_parallel(self.model_1, feature_x1, self.gpu_ids)
            feature_x4 = nn.parallel.data_parallel(self.model_2, feature_x2, self.gpu_ids)
            output_for_x4 = nn.parallel.data_parallel(self.model_3, feature_x4, self.gpu_ids)
            output_for_x6 = nn.parallel.data_parallel(self.model_4, output_for_x4, self.gpu_ids)
            output_for_x8 = nn.parallel.data_parallel(self.model_5, output_for_x6, self.gpu_ids)
            
            output_x4 = nn.parallel.data_parallel(self.model_output_x4, output_for_x4, self.gpu_ids)
            output_x6 = nn.parallel.data_parallel(self.model_output_x6, output_for_x6, self.gpu_ids)
            output_x8 = nn.parallel.data_parallel(self.model_output_x8, output_for_x8, self.gpu_ids)
            
        else:
            feature_x1 = self.model_0(input)
            feature_x2 = self.model_1(feature_x1)
            feature_x4 = self.model_2(feature_x2)
            output_for_x4 = self.model_3(feature_x4)
            output_for_x6 = self.model_4(output_for_x4)
            output_for_x8 = self.model_5(output_for_x6)
            
            output_x4 = self.model_output_x4(output_for_x4)
            output_x6 = self.model_output_x6(output_for_x6)
            output_x8 = self.model_output_x8(output_for_x8)
            
        return feature_x1, feature_x2, feature_x4, output_x8, output_x6, output_x4


class PixelNet_Network(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, gpu_ids=[]):
        super(PixelNet_Network, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=True),
                 norm_layer(ngf, track_running_stats=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      norm_layer(ngf * mult * 2, track_running_stats=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(9):
            model += [ResnetBlock(ngf * mult, use_dropout=use_dropout)]
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=True),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]     
            
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
            
        return output


class DepixelNet_Network(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, gpu_ids=[]):
        super(DepixelNet_Network, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        model_0 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=True),
                    norm_layer(ngf, track_running_stats=True),
                    nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=True),
                        norm_layer(ngf * mult * 2, track_running_stats=True),
                        nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(9):
            model_0 += [ResnetBlock(ngf * mult, use_dropout=use_dropout)] 

        mult = 4
        model_1 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=True),
                    norm_layer(int(ngf * mult / 2), track_running_stats=True),
                    nn.ReLU(True)]
        
        mult = 2
        model_2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=True),
                    norm_layer(int(ngf * mult / 2), track_running_stats=True),
                    nn.ReLU(True)]      
            
        model_3 = [nn.ReflectionPad2d(3)]
        model_3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_3 += [nn.Tanh()]

        self.model_0 = nn.Sequential(*model_0)
        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            feature_x4 = nn.parallel.data_parallel(self.model_0, input, self.gpu_ids)
            feature_x2 = nn.parallel.data_parallel(self.model_1, feature_x4, self.gpu_ids)
            feature_x1 = nn.parallel.data_parallel(self.model_2, feature_x2, self.gpu_ids)
            output = nn.parallel.data_parallel(self.model_3, feature_x1, self.gpu_ids)
            
        else:
            feature_x4 = self.model_0(input)
            feature_x2 = self.model_1(feature_x4)
            feature_x1 = self.model_2(feature_x2)
            output = self.model_3(feature_x1)
            
        return feature_x1, feature_x2, feature_x4, output


class ResnetBlock(nn.Module):
    def __init__(self, channel, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(channel, use_dropout)

    def build_conv_block(self, channel, use_dropout):
        conv_block = []
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0, bias=True),
                       norm_layer(channel, track_running_stats=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0, bias=True),
                       norm_layer(channel, track_running_stats=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, use_sigmoid=False, gpu_ids=[]):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=2, padding=padw, bias=True),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**3, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                               kernel_size=kw, stride=1, padding=padw, bias=True),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ImageGradient(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(ImageGradient, self).__init__()
        self.gpu_ids = gpu_ids

        a = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        if len(self.gpu_ids) > 0:
            conv1.weight = nn.Parameter(torch.from_numpy(a).cuda(self.gpu_ids[0]).float(), requires_grad=False)
        else:
            conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)

        b = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        if len(self.gpu_ids) > 0:
            conv2.weight = nn.Parameter(torch.from_numpy(b).cuda(self.gpu_ids[0]).float(), requires_grad=False)
        else:
            conv2.weight = nn.Parameter(torch.from_numpy(b).float(), requires_grad=False)

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            G_x = nn.parallel.data_parallel(self.conv1, input, self.gpu_ids)
            G_y = nn.parallel.data_parallel(self.conv2, input, self.gpu_ids)
        else:
            G_x = self.conv1(input)
            G_y = self.conv2(input)
        
        return G_x, G_y