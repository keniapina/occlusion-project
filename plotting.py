import numpy as np
import matplotlib.pyplot as plt
import math, os
from scipy.io import loadmat 
from sklearn.feature_extraction.image import PatchExtractor
from utils import tile_raster_images as tri
import argparse
import cPickle

parser = argparse.ArgumentParser(description='make_plots.')
parser.add_argument('file_name', type=str,
                    help='file to plot')
parser.add_argument('plot_folder', type=str,
                    help='folder for saving plots')
args = parser.parse_args()
plot_folder = args.plot_folder

file_name = args.file_name 
data = loadmat("IMAGES.mat")
ims = np.transpose(data['IMAGES'], (2,0,1))
shape = (8,8)
dim = np.prod(shape)

with open(file_name) as f:
    model = cPickle.load(f)

sparse_coding = model['sparse_coding']
w = model['w']
cost_array = model['cost_array']
recon_array = model['recon_array']
coeff_array = model['coeff_array']
l0_array = model['l0_array']
snr_array = model['snr_array']
hist_array = model['hist_array']
im_array = model['im_array']
w_array = model['w_array']

epochs, n_iter = cost_array.shape

if sparse_coding:
    sub = 'sc'
else:
    sub = 'mca'  

plot_epoch = np.linspace(0,epochs-1,4,dtype = int)

patches = PatchExtractor(shape, 64*64).transform(ims)
patches = patches.reshape(-1,dim).astype('float32')
im = tri(patches[:100], shape, (10, 10), (2,2))
plt.imshow(im, cmap='gray', interpolation='nearest')
plt.savefig(os.path.join(plot_folder, sub, 'patchces.png'))

for idx in plot_epoch:
    plt.figure()
    im = tri(w_array[idx, :64], shape, (8, 8), (2,2))
    plt.imshow(im, cmap='gray', interpolation='nearest')
    plt.savefig(os.path.join(plot_folder, sub, 'w_'+str(idx)+'.png'))

plt.figure()
for idx in plot_epoch:
    plt.plot (np.arange(n_iter),cost_array[idx], label = idx)
plt.legend (loc = 'best')
plt.title('Total Cost')
plt.savefig(os.path.join(plot_folder, sub, 'total_cost.png'))

plt.figure()
for idx in plot_epoch:
    plt.plot (np.arange(n_iter),recon_array[idx], label = idx)
plt.legend (loc = 'best')
plt.title('Reconstruction')
plt.savefig(os.path.join(plot_folder, sub, 'reconstruction.png'))

plt.figure()
for idx in plot_epoch:
    plt.plot (np.arange(n_iter),coeff_array[idx], label = idx)
plt.legend (loc = 'best')
plt.title('Coefficient Cost')
plt.savefig(os.path.join(plot_folder, sub, 'coefficient_cost.png'))

plt.figure()
for idx in plot_epoch:
    plt.plot (np.arange(n_iter),l0_array[idx], label = idx)
plt.legend (loc = 'best')
plt.title('L0')
plt.savefig(os.path.join(plot_folder, sub, 'L0.png'))

plt.figure()
for idx in plot_epoch:
    plt.plot (np.arange(n_iter),snr_array[idx], label = idx)
plt.legend (loc = 'best')
plt.title('SNR')
plt.savefig(os.path.join(plot_folder, sub, 'snr.png'))

plt.figure()
for idx in plot_epoch:
    plt.plot (np.arange(n_iter),10*np.log10(snr_array[idx]), label = idx)
plt.legend (loc = 'best')
plt.title('SNR(dB)')
plt.savefig(os.path.join(plot_folder, sub, 'snr_dB.png'))

for idx in plot_epoch:
    plt.figure()
    plt.hist(hist_array[idx], 25, normed=1, facecolor='gray')
    plt.savefig(os.path.join(plot_folder, sub, 'histogram_'+str(idx)+'.png'))

plt.close('all')
