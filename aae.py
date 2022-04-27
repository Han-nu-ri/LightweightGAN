import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Encoder
from models import Decoder
from models import FeatureDiscriminator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def density_compare_fig(encoder, dataloader, device) : 
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_list=[]
    for data in  dataloader: 
        encoder.eval()
        with torch.no_grad() : 
            data_cuda = data.to(device)
            feature1024_4_4 =  encoder(data_cuda)
            feature_list.append(feature1024_4_4.detach().cpu())
        if dataloader.batch_size * len(feature_list) >= 1024 : break
    feature = torch.cat(feature_list)
    index = torch.sort(abs(feature.mean(dim=(0))).view(-1), descending=True)[1][:8]
    pick_feature = feature.view(1024,-1)[:,index]
    fig, ax = plt.subplots(8, figsize=(4,32))
    for each_dim in range(pick_feature.size(1)) : 
        sns.kdeplot(pick_feature[:,each_dim], label="E(x)", ax=ax[each_dim])
        sns.kdeplot(torch.randn(1024), label='Z', ax=ax[each_dim])
        ax[each_dim].legend()
        ax[each_dim].set_title("mean=%f. sig=%f" % (pick_feature[:,each_dim].mean(),pick_feature[:,each_dim].std()) )
    return fig

def calculate_lpip_sum_given_images(loss_fn, group_of_images, device):
    group_of_images = group_of_images.to(device)
    num_rand_outputs = len(group_of_images)
    lpips_sum = loss_fn(group_of_images[0:num_rand_outputs - 1], group_of_images[1:num_rand_outputs])
    return float(lpips_sum.sum())

def calculate_lpips(group_of_images, device):
    lpips_sum = 0
    loss_fn = lpips.LPIPS(net='alex').to(device)
    for i in range(0, len(group_of_images), 8):
        if i == 0:
            start_index = i
        else:
            start_index = i - 1
        end_index = start_index + 8
        if end_index > len(group_of_images):
            end_index = len(group_of_images)
        lpips_sum += calculate_lpip_sum_given_images(loss_fn, group_of_images[start_index:end_index], device)
    return lpips_sum / (len(group_of_images)-1)


noise_dim = 256
cuda = 2
im_size=512
device = torch.device('cuda:%d'%(cuda))

decoder = Decoder( ngf=64, nz=noise_dim, nc=3, im_size=im_size)#, big=args.big )
decoder.to(device)
encoder = Encoder(ndf=64, nc=3, im_size=im_size)
encoder.to(device)
discriminator = FeatureDiscriminator().to(device)

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torchvision.datasets import DatasetFolder
import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from operation import MultiEpochsDataLoader, CudaDataLoader
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
import wandb

transform_list = [
        transforms.Resize((int(im_size),int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
trans = transforms.Compose(transform_list)


data_root = 'ffhq1024/image'
batch_size = 16
dataloader_workers=16


if 'lmdb' in data_root:
    from operation import MultiResolutionDataset
    load_size = 1024
    if im_size<=512:
        load_size = 512
    dataset = MultiResolutionDataset(data_root, trans, load_size)
else:
    dataset = ImageFolder(root=data_root, transform=trans)
    #dataset = DatasetFolder(root=data_root, transform=trans)

len(dataset)

dataloader =DataLoader(dataset, batch_size=batch_size, shuffle=True,
                  num_workers=dataloader_workers, pin_memory=True)

import generative_model_score
import importlib
importlib.reload(generative_model_score)
score_model = generative_model_score.GenerativeModelScore(dataloader)
score_model.lazy_mode(True) 

import models

score_model.load_or_gen_realimage_info(device)

param = list(encoder.parameters()) + list(decoder.parameters())
optim = torch.optim.SGD(param, lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3, verbose=True)

optim_for_dis = torch.optim.SGD(discriminator.parameters(), lr=1e-4)

import matplotlib.pyplot as plt

wandb.init(project='AE')

import pytorch_msssim
import lpips
loss_fn = lpips.LPIPS(net='alex').to(device)

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()

EPS = 1e-5
epochs = 10000
recon_list = []
for epoch in range(epochs) : 
    plt.close()
    sum_ssim_loss = 0
    sum_mse_loss = 0
    sum_per_loss = 0.
    sum_dis_loss_for_dis = 0.
    sum_dis_loss_for_en = 0.
    pbar = tqdm(dataloader, desc='AutoEncoder')
    encoder.train()
    decoder.train()
    for data in  pbar: 
        data_cuda = data.to(device)
       
        with torch.enable_grad() : 
            # train dis
            feature1024_4_4 =  encoder(data_cuda)
            dis_for_encoded = discriminator(feature1024_4_4)
            dis_for_gaussian = discriminator(torch.randn(data_cuda.size(0),1024,4,4, device=device))
            dis_loss = -torch.mean(torch.log(dis_for_gaussian+EPS)+torch.log(1-dis_for_encoded+EPS)) #가장 작은 값은 log 0. 
                        # maximize mean.      maximize gaussian.                minimize encoded
            optim_for_dis.zero_grad()
            dis_loss.backward()
            optim_for_dis.step()
            sum_dis_loss_for_dis += dis_loss.item()
        
        
        
        #train AE
        with torch.enable_grad() : 
            feature1024_4_4 =  encoder(data_cuda)
            dis_for_encoded = discriminator(feature1024_4_4)
            decoded_image =  decoder(feature1024_4_4)
            #ssim_loss = -pytorch_msssim.msssim(decoded_image, data_cuda)
            ssim_loss = torch.tensor([0.])
            mse_loss = l1(decoded_image, data_cuda)
            percent_loss =  loss_fn.forward(decoded_image, data_cuda).mean()
            #percent_loss = torch.tensor([0.])
            dis_loss = -torch.mean(torch.log(dis_for_encoded+EPS))
                       #maximize mean. maximize encoded

            loss = dis_loss #ssim_loss + mse_loss + percent_loss + dis_loss
            #loss = mse_loss+ dis_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            sum_ssim_loss += ssim_loss.item()
            sum_mse_loss += mse_loss.item()
            sum_per_loss += percent_loss.item()
            sum_dis_loss_for_en += dis_loss.item()
        pbar.set_description("[%d/%d, (ssim, l1, perc)=%f,%f,%f / l=%f]" % \
                             (epoch, epochs, sum_ssim_loss, sum_mse_loss, sum_per_loss, loss.item()))
     
    scheduler.step(sum_mse_loss)
               
        
    with torch.no_grad() : 
        fig, ax = plt.subplots(8,3, figsize=(8,32))
        fig.suptitle(str(epoch))
        gen_image = decoder(torch.randn(8,1024,4,4, device=device))
        for row in range(8) : 
            ax[row][0].imshow(data[row].permute(1,2,0))
            ax[row][1].imshow(decoded_image[row].detach().cpu().permute(1,2,0))
            ax[row][2].imshow(gen_image[row].detach().cpu().permute(1,2,0))
            
    metrics={}
    if epoch % 10 == 0 :
        fake_images_list = []
        encoder.eval()
        decoder.eval()
        
        for data in tqdm(dataloader, desc="[Generative Score]gen fake image...") :
            with torch.no_grad() : 
                feature1024_4_4_gaussian = torch.randn(data.size(0),1024,4,4, device=device)
                decoded_image =  decoder(feature1024_4_4_gaussian)
                fake_images_list.append(decoded_image.cpu())
        
        encoder.to('cpu')
        decoder.to('cpu') 
        loss_fn.to('cpu')
        
        fake_image_tensor = torch.cat(fake_images_list)
        score_model.model_to(device)
        score_model.lazy_forward(fake_forward=True, fake_image_tensor=fake_image_tensor, device=device)
        score_model.model_to('cpu')
        encoder.to(device)
        decoder.to(device)
        loss_fn.to(device)
        score_model.calculate_fake_image_statistics()
        score_model.calculate_fake_image_statistics()
        metrics = score_model.calculate_generative_score()
        metrics['lpips'] = calculate_lpips(fake_image_tensor, device)
        
        torch.save(encoder.state_dict(), "model_pth/aae/%d_encoder.pth" % epoch)
        torch.save(decoder.state_dict(), "model_pth/aae/%d_decoder.pth" % epoch)
       
    density_fig = density_compare_fig(encoder, dataloader, device)
    log_dict = {'sum_ssim_loss' : sum_ssim_loss,
                   'sum_l1_loss' : sum_mse_loss,
                    'sum_per_loss':sum_per_loss,
                   'fig' :  wandb.Image(fig),
                    'density_fig' : wandb.Image(density_fig),
                    'lr' : scheduler._last_lr[0],
                    'sum_dis_loss_for_dis' : sum_dis_loss_for_dis,
                    'sum_dis_loss_for_en' : sum_dis_loss_for_en,
                    #'dis_for_encoded' : dis_for_encoded.detach().cpu()
                 }
    log_dict.update(metrics)
    wandb.log(log_dict)
    plt.close()


