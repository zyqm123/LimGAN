from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import plotly.io as pio
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

pio.renderers.default = "browser"


class MyData(Dataset):
    """
    Custom Dataset class to wrap the dataset and labels.
    """
    def __init__(self, dataset):
        self.imgs, self.labels = dataset

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.
        """
        img = self.imgs[idx]
        label = []
        return img, label

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.imgs)

'''
Argument parser for configuring the script with various options
'''
parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', default='./data/',required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default='123', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
parser.add_argument('--num_discriminators', type=int, default=18, help='node number of discriminator for ensemble learning')

opt = parser.parse_args()
print(opt)

# Setup output folder, seed, and device
outf = opt.outf
try:
    os.makedirs(outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

use_mps = opt.mps and torch.backends.mps.is_available()
if opt.cuda:
    device = torch.device("cuda:0")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load dataset
dataset = datasets.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
nc = 3
assert dataset

#Setup dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                         num_workers=int(opt.workers))
'''
Define network architecture
'''
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

def weights_init(m):
    """
        Custom weights initialization for Conv and BatchNorm layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class Generator(nn.Module):
    """
    Generator class for the GAN, takes random noise as input and generates images.
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        """
        Forward pass through the generator.
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    """
    Discriminator class for the GAN, takes images as input and outputs whether they are real or fake.
    """
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Forward pass through the discriminator.
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

'''
Define and train the model
'''
# Initialize Generator and Discriminator
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Initialize an ensemble of Discriminators
netD_lis = []
for _ in range(opt.num_discriminators):
    netD_tmp = Discriminator(ngpu).to(device)
    netD_tmp.apply(weights_init)
    netD_lis.append(netD_tmp)

# 2.定义损失函数
criterion = nn.BCELoss()
# 2.Define loss function and optimizers
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_lis = []
for D_i in range(opt.num_discriminators):
    optimizerD_tmp = optim.Adam(netD_lis[D_i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD_lis.append(optimizerD_tmp)

real_label = 1
fake_label = 0
if opt.dry_run:
    opt.niter = 1

# Training Loop
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        errD_lis = []
        D_x_sum = []
        D_G_z1_sum = []
        ii = 0
        for netD_i in netD_lis:
            fixed_noise = torch.randn(50, nz, 1, 1, device=device)
            data_sam_1_i = MyData(data)
            dataloader_2 = torch.utils.data.DataLoader(data_sam_1_i, batch_size=50,
                                                       shuffle=True, num_workers=int(0))

            ij = 1
            for data_ij in dataloader_2:
                if ij > 1:
                    break
                data_sam_ij = data_ij
                ij = ij + 1

            netD_i.zero_grad()
            real_cpu = data_sam_ij[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)

            output = netD_i(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            D_x_sum.append(D_x)

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD_i(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            D_G_z1_sum.append(D_G_z1)
            errD = errD_real + errD_fake
            optimizerD_lis[ii].step()
            errD_lis.append(errD)
            ii += 1
        # softmax of the output of sub Discriminators
        errD_avg = torch.tensor(errD_lis)
        D_x_sum = torch.tensor(D_x_sum)
        D_G_z1_sum = torch.tensor(D_G_z1_sum)

        softmax_2 = nn.Softmax(dim=-1)

        errD_weight = softmax_2(errD_avg)
        errD_tmp = torch.mul(errD_avg, errD_weight)
        errD_end = errD_tmp.mean(dim=-1)

        D_x_sum_weight = softmax_2(D_x_sum)
        D_x_sum_tmp = torch.mul(D_x_sum, D_x_sum_weight)
        D_x_avg = D_x_sum_tmp.mean(dim=-1)

        D_G_z1_sum_weight = softmax_2(D_G_z1_sum)
        D_G_z1_sum_tmp = torch.mul(D_G_z1_sum, D_G_z1_sum_weight)
        D_G_z1_avg = D_G_z1_sum_tmp.mean(dim=-1)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        softmax_1 = nn.Softmax(dim=0)
        output_lis = []
        for netD_tmp in netD_lis:
            output_lis.append(netD_tmp(fake))
        output_lis = torch.stack(output_lis, 0)
        output_weight = softmax_1(output_lis)
        output_tmp = torch.mul(output_lis, output_weight)
        output = output_tmp.mean(dim=0)

        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD_end.item(), errG.item(), D_x_avg, D_G_z1_avg, D_G_z2))

        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % opt.outf,
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)
        if opt.dry_run:
            break
    # Save model checkpoints
    if epoch==opt.niter-1:
        torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
        for Dsave_i in range(opt.num_discriminators):
            torch.save(netD_lis[Dsave_i].state_dict(), '%s/netD_num_%d.pth' % (outf, Dsave_i))





