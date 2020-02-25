import torch
from collections import OrderedDict
import itertools
from .image_pool import ImagePool
from . import networks
import numpy as np
from PIL import Image
import torch.nn as nn
import os
from torch.optim import lr_scheduler



class PixelizationModel():
    def name(self):
        return 'PixelizationModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt.checkpoints_dir
        
        self.gridnet = networks.define_gridnet(3, 3,  opt.ngf, not opt.no_dropout, self.gpu_ids)
        
        self.pixelnet = networks.define_pixelnet(3, 3, opt.ngf, not opt.no_dropout, self.gpu_ids)        
        
        self.depixelnet = networks.define_depixelnet(3, 3, opt.ngf, not opt.no_dropout, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_gridnet = networks.define_D(3, opt.ndf, use_sigmoid, self.gpu_ids)
            
            self.netD_pixelnet = networks.define_D(3, opt.ndf, use_sigmoid, self.gpu_ids)            
            
            self.netD_depixelnet = networks.define_D(3, opt.ndf, use_sigmoid, self.gpu_ids)

            self.get_gradient = networks.ImageGradient(self.gpu_ids)
        
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.gridnet, 'gridnet', which_epoch)
            self.load_network(self.pixelnet, 'pixelnet', which_epoch)
            self.load_network(self.depixelnet, 'depixelnet', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_gridnet, 'netD_gridnet', which_epoch)
                self.load_network(self.netD_pixelnet, 'netD_pixelnet', which_epoch)
                self.load_network(self.netD_depixelnet, 'netD_depixelnet', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_small_x8_pool = ImagePool(opt.pool_size)
            self.fake_B_small_x6_pool = ImagePool(opt.pool_size)
            self.fake_B_small_x4_pool = ImagePool(opt.pool_size)
            self.fake_B_x8_processed_pool = ImagePool(opt.pool_size)
            self.fake_B_x6_processed_pool = ImagePool(opt.pool_size)
            self.fake_B_x4_processed_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            self.optimizer_gridnet = torch.optim.Adam(self.gridnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_pixelnet = torch.optim.Adam(self.pixelnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_depixelnet = torch.optim.Adam(self.depixelnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizer_D_gridnet = torch.optim.Adam(self.netD_gridnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_pixelnet = torch.optim.Adam(self.netD_pixelnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_depixelnet = torch.optim.Adam(self.netD_depixelnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_gridnet)
            self.optimizers.append(self.optimizer_pixelnet)
            self.optimizers.append(self.optimizer_D_gridnet)
            self.optimizers.append(self.optimizer_D_pixelnet)            
            self.optimizers.append(self.optimizer_D_depixelnet)
            for optimizer in self.optimizers:
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
                self.schedulers.append(scheduler)

        print('---------- Networks initialized -------------')
        networks.print_network(self.gridnet)
        networks.print_network(self.pixelnet)
        networks.print_network(self.depixelnet)
        if self.isTrain:
            networks.print_network(self.netD_gridnet)
            networks.print_network(self.netD_pixelnet)
            networks.print_network(self.netD_depixelnet)
        print('-----------------------------------------------')
        
        
    def change_size(self, img, size=None, ratio=None):
        img = img.detach()
        if size:
            img = torch.nn.functional.interpolate(img, size=size, mode='nearest')
        elif ratio:
            if ratio == 6:
                img = torch.nn.functional.interpolate(img, scale_factor=(3.0/16.0), mode='nearest')
            else:
                img = torch.nn.functional.interpolate(img, scale_factor=(1.0/ratio), mode='nearest')
        
        return img
        
    def set_input(self, input):
        # A is the original clip art we want to pixelize
        # B is the original pixel art we want to depixelize
        input_A = input['A']
        input_A_small_x8 = self.change_size(input['A'], ratio=8)
        input_A_small_x6 = self.change_size(input['A'], ratio=6)
        input_A_small_x4 = self.change_size(input['A'], ratio=4)
        
        input_B = input['B']
        input_B_small_x8 = self.change_size(input['B'], ratio=8)
        input_B_small_x6 = self.change_size(input['B'], ratio=6)
        input_B_small_x4 = self.change_size(input['B'], ratio=4)
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_A_small_x8 = input_A_small_x8.cuda(self.gpu_ids[0], async=True)
            input_A_small_x6 = input_A_small_x6.cuda(self.gpu_ids[0], async=True) 
            input_A_small_x4 = input_A_small_x4.cuda(self.gpu_ids[0], async=True)                                         
            
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_B_small_x8 = input_B_small_x8.cuda(self.gpu_ids[0], async=True)
            input_B_small_x6 = input_B_small_x6.cuda(self.gpu_ids[0], async=True)   
            input_B_small_x4 = input_B_small_x4.cuda(self.gpu_ids[0], async=True)           
            
        self.real_A = input_A
        self.real_A_small_x8 = input_A_small_x8
        self.real_A_small_x6 = input_A_small_x6
        self.real_A_small_x4 = input_A_small_x4
        
        self.real_B = input_B
        self.real_B_small_x8 = input_B_small_x8
        self.real_B_small_x6 = input_B_small_x6
        self.real_B_small_x4 = input_B_small_x4
        self.image_paths = input['A_paths']

    def test(self):
        # Original clip art as input
        real_A = self.real_A
        size_A = real_A[0][0].shape
        _, _, _, fake_B_small_x8, fake_B_small_x6, fake_B_small_x4 = self.gridnet(real_A)

        # Upsample the output of GridNet to the original resolution (By Nearest Interpolation)
        fake_B_x8 = self.change_size(fake_B_small_x8, size=size_A)
        fake_B_x6 = self.change_size(fake_B_small_x6, size=size_A)
        fake_B_x4 = self.change_size(fake_B_small_x4, size=size_A)
        
        # Final our pixel art
        _, _, _, fake_B_x8_processed = self.pixelnet(fake_B_x8)
        _, _, _, fake_B_x6_processed = self.pixelnet(fake_B_x6)
        _, _, _, fake_B_x4_processed = self.pixelnet(fake_B_x4)
        
        self.fake_B_x8_processed = fake_B_x8_processed.data
        self.fake_B_x6_processed = fake_B_x6_processed.data
        self.fake_B_x4_processed = fake_B_x4_processed.data

        # Original pixel art as input
        real_B = self.real_B
        size_B = real_B[0][0].shape

        # Final our clip art
        _, _, _, fake_A = self.depixelnet(real_B)

        self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # Discriminator in GridNet
    def backward_D_gridnet(self):
        fake_B_small_x8 = self.fake_B_small_x8_pool.query(self.fake_B_small_x8)
        fake_B_small_x6 = self.fake_B_small_x6_pool.query(self.fake_B_small_x6)
        fake_B_small_x4 = self.fake_B_small_x4_pool.query(self.fake_B_small_x4)        
        loss_D_gridnet = self.backward_D_basic(self.netD_gridnet, self.real_B_small_x8.float(), fake_B_small_x8)
        loss_D_gridnet += self.backward_D_basic(self.netD_gridnet, self.real_B_small_x6.float(), fake_B_small_x6)
        loss_D_gridnet += self.backward_D_basic(self.netD_gridnet, self.real_B_small_x4.float(), fake_B_small_x4)
        
        self.loss_D_gridnet = loss_D_gridnet.item()

    # Discriminator in PixelNet  
    def backward_D_pixelnet(self):
        fake_B_x8_processed = self.fake_B_x8_processed_pool.query(self.fake_B_x8_processed)
        fake_B_x6_processed = self.fake_B_x6_processed_pool.query(self.fake_B_x6_processed)
        fake_B_x4_processed = self.fake_B_x4_processed_pool.query(self.fake_B_x4_processed)
        loss_D_pixelnet = self.backward_D_basic(self.netD_pixelnet, self.real_B.float(), fake_B_x8_processed)
        loss_D_pixelnet += self.backward_D_basic(self.netD_pixelnet, self.real_B.float(), fake_B_x6_processed)
        loss_D_pixelnet += self.backward_D_basic(self.netD_pixelnet, self.real_B.float(), fake_B_x4_processed)
        self.loss_D_pixelnet = loss_D_pixelnet.item()

    # Discriminator in DepixelNet
    def backward_D_depixelnet(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_depixelnet = self.backward_D_basic(self.netD_depixelnet, self.real_A, fake_A)
        self.loss_D_depixelnet = loss_D_depixelnet.item()

    def backward_all_net(self):
        lambda_weight = self.opt.lambda_weight
        size_A = self.real_A[0][0].shape
        size_B = self.real_B[0][0].shape
     
        # The whole losses of GridNet
        # Adversarial loss of GridNet
        feature_x1_gridnet, feature_x2_gridnet, feature_x4_gridnet, fake_B_small_x8, fake_B_small_x6, fake_B_small_x4 = self.gridnet(self.real_A)
        pred_fake_x8 = self.netD_gridnet(fake_B_small_x8)
        pred_fake_x6 = self.netD_gridnet(fake_B_small_x6)
        pred_fake_x4 = self.netD_gridnet(fake_B_small_x4)        
        
        loss_gridnet_adversarial = self.criterionGAN(pred_fake_x8, True)
        loss_gridnet_adversarial += self.criterionGAN(pred_fake_x6, True)
        loss_gridnet_adversarial += self.criterionGAN(pred_fake_x4, True)

        # L1 loss of GridNet
        loss_gridnet_IO_L1 = self.criterionL1(fake_B_small_x8, self.real_A_small_x8.float())
        loss_gridnet_IO_L1 += self.criterionL1(fake_B_small_x6, self.real_A_small_x6.float())
        loss_gridnet_IO_L1 += self.criterionL1(fake_B_small_x4, self.real_A_small_x4.float())
        loss_gridnet_IO_L1 *= lambda_weight
        
        # Gradient loss of GridNet
        fake_B_small_x8_gradient_x, fake_B_small_x8_gradient_y = self.get_gradient(fake_B_small_x8)
        fake_B_small_x6_gradient_x, fake_B_small_x6_gradient_y = self.get_gradient(fake_B_small_x6)
        fake_B_small_x4_gradient_x, fake_B_small_x4_gradient_y = self.get_gradient(fake_B_small_x4)
        real_A_small_x8_gradient_x, real_A_small_x8_gradient_y = self.get_gradient(self.real_A_small_x8.float())
        real_A_small_x6_gradient_x, real_A_small_x6_gradient_y = self.get_gradient(self.real_A_small_x6.float())
        real_A_small_x4_gradient_x, real_A_small_x4_gradient_y = self.get_gradient(self.real_A_small_x4.float())

        loss_gridnet_G_L1_x8 = self.criterionL1(fake_B_small_x8_gradient_x, real_A_small_x8_gradient_x) \
                             + self.criterionL1(fake_B_small_x8_gradient_y, real_A_small_x8_gradient_y)
        loss_gridnet_G_L1_x6 = self.criterionL1(fake_B_small_x6_gradient_x, real_A_small_x6_gradient_x) \
                             + self.criterionL1(fake_B_small_x6_gradient_y, real_A_small_x6_gradient_y)
        loss_gridnet_G_L1_x4 = self.criterionL1(fake_B_small_x4_gradient_x, real_A_small_x4_gradient_x) \
                             + self.criterionL1(fake_B_small_x4_gradient_y, real_A_small_x4_gradient_y)

        loss_gridnet_G_L1 = (loss_gridnet_G_L1_x8 + loss_gridnet_G_L1_x6 + loss_gridnet_G_L1_x4)
        loss_gridnet_G_L1 *= 0.5

        '''
        # Mirror loss of GridNet
        loss_mirror_B_1 = self.criterionL1(rec_B_small_x8, self.real_B_small_x8.float())
        loss_mirror_B_1 += self.criterionL1(rec_B_small_x6, self.real_B_small_x6.float())
        loss_mirror_B_1 += self.criterionL1(rec_B_small_x4, self.real_B_small_x4.float())
        loss_mirror_B_1 *= lambda_weight / 2
        '''
        
        self.loss_gridnet = loss_gridnet_adversarial + loss_gridnet_IO_L1 + loss_gridnet_G_L1
        
        # The whole losses of PixelNet
        # Adversarial loss of PixelNet
        fake_B_x8 = self.change_size(fake_B_small_x8.data, size=size_A)
        _, _, _, fake_B_x8_processed = self.pixelnet(fake_B_x8)
        pred_fake_x8 = self.netD_pixelnet(fake_B_x8_processed)
        
        fake_B_x6 = self.change_size(fake_B_small_x6.data, size=size_A)
        _, _, _, fake_B_x6_processed = self.pixelnet(fake_B_x6)        
        pred_fake_x6 = self.netD_pixelnet(fake_B_x6_processed)        
        
        fake_B_x4 = self.change_size(fake_B_small_x4.data, size=size_A)
        _, _, _, fake_B_x4_processed = self.pixelnet(fake_B_x4)
        pred_fake_x4 = self.netD_pixelnet(fake_B_x4_processed)        
        
        loss_pixelnet_adversarial = self.criterionGAN(pred_fake_x8, True)
        loss_pixelnet_adversarial += self.criterionGAN(pred_fake_x6, True)
        loss_pixelnet_adversarial += self.criterionGAN(pred_fake_x4, True)

        # L1 loss of PixelNet
        loss_pixelnet_IO_L1 = self.criterionL1(fake_B_x8_processed, fake_B_x8)
        loss_pixelnet_IO_L1 += self.criterionL1(fake_B_x6_processed, fake_B_x6)
        loss_pixelnet_IO_L1 += self.criterionL1(fake_B_x4_processed, fake_B_x4)
        loss_pixelnet_IO_L1 *= lambda_weight / 20.0
        
        # Gradient loss of PixelNet
        fake_B_x8_processed_gradient_x, fake_B_x8_processed_gradient_y = self.get_gradient(fake_B_x8_processed)
        fake_B_x6_processed_gradient_x, fake_B_x6_processed_gradient_y = self.get_gradient(fake_B_x6_processed)
        fake_B_x4_processed_gradient_x, fake_B_x4_processed_gradient_y = self.get_gradient(fake_B_x4_processed)
        fake_B_x8_gradient_x, fake_B_x8_gradient_y = self.get_gradient(fake_B_x8)
        fake_B_x6_gradient_x, fake_B_x6_gradient_y = self.get_gradient(fake_B_x6)
        fake_B_x4_gradient_x, fake_B_x4_gradient_y = self.get_gradient(fake_B_x4)
        loss_pixelnet_G_L1_x8 = self.criterionL1(fake_B_x8_processed_gradient_x, fake_B_x8_gradient_x) \
                              + self.criterionL1(fake_B_x8_processed_gradient_y, fake_B_x8_gradient_y)
        loss_pixelnet_G_L1_x6 = self.criterionL1(fake_B_x6_processed_gradient_x, fake_B_x6_gradient_x) \
                              + self.criterionL1(fake_B_x6_processed_gradient_y, fake_B_x6_gradient_y)
        loss_pixelnet_G_L1_x4 = self.criterionL1(fake_B_x4_processed_gradient_x, fake_B_x4_gradient_x) \
                              + self.criterionL1(fake_B_x4_processed_gradient_y, fake_B_x4_gradient_y)
        loss_pixelnet_G_L1 = (loss_pixelnet_G_L1_x8 + loss_pixelnet_G_L1_x6 + loss_pixelnet_G_L1_x4)
        loss_pixelnet_G_L1 *= 0.5

        # Mirror loss of PixelNet
        
        _, _, _, fake_A = self.depixelnet(self.real_B)
        '''
        _, _, _, rec_B_small_x8, rec_B_small_x6, rec_B_small_x4 = self.gridnet(fake_A)

        rec_B_x8 = self.change_size(rec_B_small_x8.data, size=size_B)
        _, _, _, rec_B_x8_processed = self.pixelnet(rec_B_x8)
        
        rec_B_x6 = self.change_size(rec_B_small_x6.data, size=size_B)
        _, _, _, rec_B_x6_processed = self.pixelnet(rec_B_x6)        
        
        rec_B_x4 = self.change_size(rec_B_small_x4.data, size=size_B)
        _, _, _, rec_B_x4_processed = self.pixelnet(rec_B_x4)
        
        loss_mirror_B_2 = (self.criterionL1(rec_B_x8_processed, self.real_B), 
                           self.criterionL1(rec_B_x6_processed, self.real_B), 
                           self.criterionL1(rec_B_x4_processed, self.real_B))
        index = np.argmin((loss_mirror_B_2[0].item(), loss_mirror_B_2[1].item(), loss_mirror_B_2[2].item()))
        loss_mirror_B_2 = loss_mirror_B_2[index]
        loss_mirror_B_2 *= lambda_weight / 10
        '''
        #  + loss_mirror_B_2
        
        self.loss_pixelnet = loss_pixelnet_adversarial + loss_pixelnet_IO_L1 + loss_pixelnet_G_L1

        # The whole losses of DepixelNet
        # Adversarial loss of DepixelNet
        pred_fake = self.netD_depixelnet(fake_A)
        loss_depixelnet_adversarial = self.criterionGAN(pred_fake, True)

        # L1 loss of DepixelNet
        loss_depixelnet_IO_L1 = self.criterionL1(fake_A, self.real_B)
        loss_depixelnet_IO_L1 *= lambda_weight / 10.0

        # Mirror loss of DepixelNet
        depixelnet_mirror_input = (fake_B_x8_processed, fake_B_x6_processed, fake_B_x4_processed)
        feature_x1_depixelnet, feature_x2_depixelnet, feature_x4_depixelnet, rec_A = self.depixelnet(depixelnet_mirror_input[np.random.randint(0, 3)])
        loss_mirror_A = self.criterionL1(rec_A, self.real_A)
        loss_mirror_A += self.criterionL1(feature_x1_depixelnet, feature_x1_gridnet.detach()) \
                       + self.criterionL1(feature_x2_depixelnet, feature_x2_gridnet.detach()) \
                       + self.criterionL1(feature_x4_depixelnet, feature_x4_gridnet.detach())
        loss_mirror_A *= lambda_weight

        self.loss_depixelnet = loss_depixelnet_adversarial + loss_depixelnet_IO_L1 + loss_mirror_A

        self.fake_B_small_x8 = fake_B_small_x8.data
        self.fake_B_small_x6 = fake_B_small_x6.data
        self.fake_B_small_x4 = fake_B_small_x4.data
        
        # Our pixel art
        self.fake_B_x8_processed = fake_B_x8_processed.data
        self.fake_B_x6_processed = fake_B_x6_processed.data
        self.fake_B_x4_processed = fake_B_x4_processed.data
        # Our clip art
        self.fake_A = fake_A.data

        self.loss_gridnet_adversarial = loss_gridnet_adversarial.item()
        self.loss_pixelnet_adversarial = loss_pixelnet_adversarial.item()
        self.loss_depixelnet_adversarial = loss_depixelnet_adversarial.item()
        
        self.loss_gridnet_item = self.loss_gridnet.item()
        self.loss_pixelnet_item = self.loss_pixelnet.item()
        self.loss_depixelnet_item = self.loss_depixelnet.item()

    def optimize_parameters(self):
        # backward
        self.backward_all_net()        
        # Update GridNet
        self.optimizer_gridnet.zero_grad()
        self.loss_gridnet.backward(retain_graph=True)
        self.optimizer_gridnet.step()
        # Update PixelNet
        self.optimizer_pixelnet.zero_grad()
        self.loss_pixelnet.backward(retain_graph=True)
        self.optimizer_pixelnet.step()
        # depixelnet
        self.optimizer_depixelnet.zero_grad()
        self.loss_depixelnet.backward()
        self.optimizer_depixelnet.step()
        # D_gridnet
        self.optimizer_D_gridnet.zero_grad()
        self.backward_D_gridnet()
        self.optimizer_D_gridnet.step()
        # D_pixelnet
        self.optimizer_D_pixelnet.zero_grad()
        self.backward_D_pixelnet()
        self.optimizer_D_pixelnet.step()
        # D_depixelnet
        self.optimizer_D_depixelnet.zero_grad()
        self.backward_D_depixelnet()
        self.optimizer_D_depixelnet.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_gridnet', self.loss_D_gridnet), ('loss_gridnet', self.loss_gridnet_item), 
                                  ('D_pixelnet', self.loss_D_pixelnet),('loss_pixelnet', self.loss_pixelnet_item), 
                                  ('D_depixelnet', self.loss_D_depixelnet), ('loss_depixelnet', self.loss_depixelnet_item)])
        return ret_errors

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

        return image_numpy

    def get_current_visuals_train(self):
        real_A = self.tensor2im(self.real_A)
        real_B = self.tensor2im(self.real_B)

        fake_B_x8_processed = self.tensor2im(self.fake_B_x8_processed)
        fake_B_x6_processed = self.tensor2im(self.fake_B_x6_processed)
        fake_B_x4_processed = self.tensor2im(self.fake_B_x4_processed)
        fake_A = self.tensor2im(self.fake_A)
        
        ret_visuals = OrderedDict([('real_A', real_A), ('real_B', real_B), ('fake_B_x8_processed', fake_B_x8_processed),
                                   ('fake_B_x6_processed', fake_B_x6_processed), ('fake_B_x4_processed', fake_B_x4_processed), ('fake_A', fake_A), ])
        return ret_visuals
    
    def get_current_visuals_test(self):
        real_A = self.tensor2im(self.real_A)
        real_B = self.tensor2im(self.real_B)
        
        fake_B_x8_processed = self.tensor2im(self.fake_B_x8_processed)
        fake_B_x6_processed = self.tensor2im(self.fake_B_x6_processed)
        fake_B_x4_processed = self.tensor2im(self.fake_B_x4_processed)
        fake_A = self.tensor2im(self.fake_A)
        
        ret_visuals = OrderedDict([('real_A', real_A), ('real_B', real_B), ('fake_A', fake_A), ('fake_B_x4_processed', fake_B_x4_processed)])
                                #    ('fake_B_x8_processed', fake_B_x8_processed), ('fake_B_x6_processed', fake_B_x6_processed)])
        return ret_visuals

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save(self, label):
        self.save_network(self.gridnet, 'gridnet', label, self.gpu_ids)
        self.save_network(self.pixelnet, 'pixelnet', label, self.gpu_ids)
        self.save_network(self.depixelnet, 'depixelnet', label, self.gpu_ids)
        self.save_network(self.netD_gridnet, 'D_gridnet', label, self.gpu_ids)
        self.save_network(self.netD_pixelnet, 'D_pixelnet', label, self.gpu_ids)
        self.save_network(self.netD_depixelnet, 'D_depixelnet', label, self.gpu_ids)
