import argparse
import os
from tqdm import tqdm
from dcgan import Generator, Discriminator # can replace with other GAN architectures
from calibration import LRcalibrator
from en_repgan import repgan
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import numpy as np

'''
Argument parser for configuring the script with various options
'''
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./sample/', help='path to output folder for samples')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--load-g', default='./model/netG.pth', help='path for the generator file')
parser.add_argument('--load-d', default='./model/netD', help='path for the discriminator file')
parser.add_argument('--load_model', default='./model/', help='path for the discriminator and generator file')
parser.add_argument('--image-size', type=int, default=64, help='image size (input/output size for the discriminator/generator')
parser.add_argument('--ndf', type=int, default=64, help='num features in discriminator')
parser.add_argument('--ngf', type=int, default=64, help='num features in generator')
parser.add_argument('--nz', type=int, default=100, help='latent space dimensions')
parser.add_argument('--nc', type=int, default=3, help='number of image channels')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus')
parser.add_argument('--batchsize', type=int, default=100, help='batch size for sampling')
parser.add_argument('--numimages', type=int, default=5000, help='total numbers of required samples')
parser.add_argument('--calibrate', action='store_true', help='whether to calibrate the discriminator scores (if true, use LR, else, use id mapping)')
parser.add_argument('--clen', type=int, default=640, help='length of each Markov chain')
parser.add_argument('--tau', type=float, default=0.05, help='Langevin step size in L2MC')
parser.add_argument('--eta', type=float, default=0.1, help='scale of white noise (default to sqrt(tau))')

opt = parser.parse_args()

# Paths for saving models and outputs
model_path = opt.load_model
save_path = opt.outf
try:
    os.makedirs(save_path)
except OSError:
    pass

# Initialize input variables for model loading
nx = 3
nz = opt.nz
num_D = 18
batchsize = opt.batchsize

# Load the pre-trained models
load_g = opt.load_g
load_d = opt.load_d
# Select device for computation (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the pre-trained Generator model
netG = Generator(opt.ngpu, nz, opt.ngf, nx).to(device)
netG.load_state_dict(torch.load(load_g, map_location=device))

# Load the pre-trained Discriminator models (ensemble of discriminators)
netD_lis = []
for D_i in range(num_D):
    netD_tmp = Discriminator(opt.ngpu, opt.ndf, nx).to(device)
    netD_tmp.load_state_dict(
        torch.load(load_d + '_num_' + str(D_i) + '.pth', map_location=device))
    netD_lis.append(netD_tmp)

print('model loaded')
torch.set_grad_enabled(False)

# If calibration is enabled, prepare dataset and set up calibrator
if opt.calibrate:
    dataset = datasets.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True,
                                              num_workers=0)
    calibrator = LRcalibrator(netG, netD_lis, dataset, device, nz=nz)
else:
    calibrator = torch.nn.Identity()

print('start sampling')
accepted_samples = [] # List to store accepted samples
for i in tqdm(range(0, opt.numimages, opt.batchsize)):
    samples = repgan(netG, netD_lis, calibrator, device, nz, opt.batchsize, opt.clen, opt.tau, opt.eta)
    accepted_samples.append(samples.cpu())

# Save the generated samples in .npy format
accepted_samples = torch.cat(accepted_samples, dim=0).numpy()
np.save(save_path+'/repgan_samples.npy', accepted_samples)
print('samples save to repgan_samples.npy')
