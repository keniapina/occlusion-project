import numpy as np
import theano 
import theano.tensor as T
import matplotlib.pyplot as plt
import math, os
from scipy.io import loadmat 
from sklearn.feature_extraction.image import PatchExtractor
from theano.compat.python2x import OrderedDict
from utils import tile_raster_images as tri
import argparse
import cPickle

parser = argparse.ArgumentParser(description='make_plots.')
parser.add_argument('file_name', type=str,
                    help='file to save in')
args = parser.parse_args()


shape = (8,8)
sparse_coding = True
n_batch = 128
n_iter = 200
epochs = 50
dim = np.prod(shape)
n_neurons = 2*dim 
lam = .5 
eps_a = .1
eps_w = .1


data = loadmat("IMAGES.mat")

ims = np.transpose(data['IMAGES'], (2,0,1))

patches = PatchExtractor(shape, 64*64).transform(ims)
patches = patches.reshape(-1,dim).astype('float32')


if sparse_coding:
    sub = 'sc'
else:
    sub = 'mca'  



x = theano.shared(np.zeros((n_batch,dim)).astype('float32'))
w = np.random.randn(n_neurons,dim)
w_norm = np.sqrt(np.sum(w**2, axis=1, keepdims=True))
w = theano.shared((w/w_norm).astype('float32'))
a = theano.shared(np.zeros((n_batch,n_neurons)).astype('float32'))
x_batch = T.matrix() 


if sparse_coding:
    x_hat = T.dot(a,w)
else:
    x_hat = 1.
recon = (.5*(x- x_hat)**2).sum(axis=1).mean()
coeff = abs(a).sum(axis=1).mean()
l0 = T.neq(a,0.).sum(axis=1).mean()
costfunction = recon+ lam*coeff 
snr = T.mean(x.norm(2,axis=1)**2/(x-x_hat).norm(2,axis=1)**2)
grad_a = T.grad(costfunction,a)  
grad_w = T.grad(costfunction,w)
grad_recon_a = T.grad(recon,a)
grad_sparse_a = T.grad(lam*coeff, a)

updates = OrderedDict()
updates[a] = 0.*a 
zero_a = theano.function([],[],updates = updates)

updates = OrderedDict()
a_prime = a - eps_a*grad_recon_a
a_dprime = a_prime - eps_a*grad_sparse_a
m = T.eq(T.sgn(a_prime), T.sgn(a_dprime))
updates[a] = m*a_dprime
step_a = theano.function([],[costfunction,recon,coeff,l0, snr],updates = updates)

updates = OrderedDict()
updates[x] = x_batch
update_x = theano.function([x_batch],[],updates = updates)

updates = OrderedDict()
wp = w - eps_w*grad_w
wp_norm = T.sqrt(T.sum(w**2, axis=1, keepdims=True))
updates[w] = wp/wp_norm
step_w = theano.function([],[costfunction],updates = updates) 

reconstruct = theano.function([],x_hat)

cost_array = np.zeros((epochs, n_iter))
recon_array = np.zeros((epochs, n_iter))
coeff_array = np.zeros((epochs,n_iter))
l0_array = np.zeros((epochs,n_iter))
snr_array = np.zeros((epochs, n_iter))
hist_array = np.zeros((epochs, n_batch, n_neurons))

for kk in range(epochs):
    print kk
    for ii in range(int(math.floor(1.*patches.shape[0]/n_batch))):
        zero_a()
        update_x(patches[ii*n_batch : (ii+1)*n_batch])
        for jj in range(n_iter):
            cost, recon, coeff, l0, snr = step_a() 
            if ii == 0:
                cost_array[kk, jj] = cost
                recon_array[kk, jj] = recon
                coeff_array[kk, jj] = coeff
                l0_array[kk, jj] = l0
                snr_array[kk, jj] = snr
        if ii == 0:
            hist_array[kk] = a.get_value()
        step_w()

model = dict() 
model['sparse_coding']  = sparse_coding
model['w'] = w.get_value()
model['cost_array'] = cost_array
model['recon_array'] = recon_array
model['l0_array'] = l0_array
model['snr_array'] = snr_array
model['hist_array'] = hist_array
model['coeff_array'] = coeff_array

with open(args.file_name, 'w') as f:
    cPickle.dump(model, f) 
