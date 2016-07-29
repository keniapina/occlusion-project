import numpy as np
import theano 
import theano.tensor as T
import math, os
from scipy.io import loadmat 
from sklearn.feature_extraction.image import PatchExtractor
from theano.compat.python2x import OrderedDict
from utils import tile_raster_images as tri
import argparse
import cPickle



sparse_coding = True
dataset = 'images'
fista = True


shape = (8,8)
n_batch = 128
n_iter = 100
epochs = 150
decay_init = 50
decay_time = 50
p = np.exp(np.log(.1)/decay_time)
print p
dim = np.prod(shape)
n_neurons = 2*dim 
eps_a = .1
eps_w = theano.shared(np.array(.1).astype('float32'))
if sparse_coding:
    lam = .006
    file_name = 'sc.pkl'
else:
    lam = .0005
    file_name = 'mca.pkl'

if dataset == 'images':
    data = loadmat("IMAGES.mat")
    ims = np.transpose(data['IMAGES'], (2,0,1))
    patches = PatchExtractor(shape, 64*64).transform(ims)
    patches = patches.reshape(-1,dim).astype('float32')
elif dataset == 'bars':
    pass
else:
    pass

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

x_hat_sc = T.dot(a,w)
x_hat_mca = T.max(a.dimshuffle(0,1,'x')*w.dimshuffle('x',0,1), axis=1)
recon_sc = (.5*(x- x_hat_sc)**2).sum(axis=1).mean()
recon_mca = (.5*(x- x_hat_mca)**2).sum(axis=1).mean()
coeff = abs(a).sum(axis=1).mean()
l0 = T.neq(a,0.).sum(axis=1).mean()
costfunction_sc = recon_sc+ lam*coeff
costfunction_mca = recon_mca+ lam*coeff
snr_sc = T.mean(x.norm(2,axis=1)**2/(x-x_hat_sc).norm(2,axis=1)**2)
snr_mca = T.mean(x.norm(2,axis=1)**2/(x-x_hat_mca).norm(2,axis=1)**2)

if sparse_coding:
    x_hat = x_hat_sc
    costfunction = costfunction_sc
    snr = snr_sc
else:
    x_hat = x_hat_mca
    costfunction = costfunction_mca
    snr = snr_mca

if sparse_coding:
    grad_w = T.grad(costfunction_sc,w)
else:
    grad_w = T.grad(costfunction_mca, w)
grad_recon_a_sc = T.grad(recon_sc,a)
grad_recon_a_mca = T.grad(recon_mca,a)
grad_sparse_a = T.grad(lam*coeff, a)

updates = OrderedDict()
updates[eps_w] = np.array(p).astype('float32') * eps_w
decay_eps_w = theano.function([], [], updates=updates)

updates_sc = OrderedDict()
updates_mca = OrderedDict()
if fista:
    t = theano.shared(np.array(1.).astype('float32'))
    a_old = theano.shared(np.zeros((n_batch,n_neurons)).astype('float32'))

    a_prime_sc = a - eps_a * grad_recon_a_sc
    abs_a_sc = abs(a_prime_sc) - lam*eps_a
    a_ista_sc = T.sgn(a_prime_sc) * T.nnet.relu(abs_a_sc)

    a_prime_mca = a - eps_a * grad_recon_a_mca
    abs_a_mca = abs(a_prime_mca) - lam*eps_a
    a_ista_mca = T.sgn(a_prime_mca) * T.nnet.relu(abs_a_mca)

    t1 = 0.5 * (1 + T.sqrt(1. + 4. * t ** 2))
    a_new_sc = a_ista_sc + (t1 - 1.) / t * (a_ista_sc - a_old)
    a_new_mca = a_ista_mca + (t1 - 1.) / t * (a_ista_mca - a_old)

    updates_sc[a] = a_new_sc
    updates_sc[a_old] = a_ista_sc
    updates_sc[t] = t1

    updates_mca[a] = a_new_mca
    updates_mca[a_old] = a_ista_mca
    updates_mca[t] = t1
else:
    a_prime = a - eps_a*grad_recon_a
    a_dprime = a_prime - eps_a*grad_sparse_a
    m = T.eq(T.sgn(a_prime), T.sgn(a_dprime))
    updates[a] = m*a_dprime
step_a_sc = theano.function([], [costfunction_sc, recon_sc, coeff, l0, snr],
                            updates=updates_sc)
step_a_mca = theano.function([], [costfunction_mca, recon_mca, coeff, l0, snr],
                             updates=updates_mca)

updates = OrderedDict()
if fista:
    updates[a] = 0.*a 
    updates[a_old] = 0. *a_old
    updates[t] = 1.
else:
    updates[a] = 0.*a 
initialize = theano.function([],[],updates = updates)


updates = OrderedDict()
updates[x] = x_batch
update_x = theano.function([x_batch],[],updates = updates)

updates = OrderedDict()
wp = w - eps_w*grad_w
wp_norm = T.sqrt(T.sum(w**2, axis=1, keepdims=True))
updates[w] = wp/wp_norm
if sparse_coding:
    step_w = theano.function([],[costfunction],updates = updates)
else:
    step_w = theano.function([],[costfunction],updates = updates)

reconstruct = theano.function([],x_hat)
original = theano.function([], x)

cost_array = np.zeros((epochs, n_iter))
recon_array = np.zeros((epochs, n_iter))
coeff_array = np.zeros((epochs,n_iter))
l0_array = np.zeros((epochs,n_iter))
snr_array = np.zeros((epochs, n_iter))
hist_array = np.zeros((epochs, n_batch, n_neurons))
im_array = np.zeros((2, epochs, n_batch, dim))
w_array = np.zeros((epochs, n_neurons, dim))

for kk in range(epochs):
    print kk
    for ii in range(int(math.floor(1.*patches.shape[0]/n_batch))):
        initialize()
        update_x(patches[ii*n_batch : (ii+1)*n_batch])
        for jj in range(n_iter):
            if sparse_coding:
                cost, recon, coeff, l0, snr = step_a_sc()
            else:
                if jj < n_iter/10:
                    cost, recon, coeff, l0, snr = step_a_sc()
                else:
                    cost, recon, coeff, l0, snr = step_a_mca()
            if ii == 0:
                cost_array[kk, jj] = cost
                recon_array[kk, jj] = recon
                coeff_array[kk, jj] = coeff
                l0_array[kk, jj] = l0
                snr_array[kk, jj] = snr
        if kk > decay_init:
             decay_eps_w()
        if ii == 0:
            hist_array[kk] = a.get_value()
            im_array[0, kk] = np.array(original())
            im_array[1, kk] = reconstruct()
            w_array[kk] = w.get_value()
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
model['im_array'] = im_array
model['w_array'] = w_array
with open(args.file_name, 'w') as f:
    cPickle.dump(model, f) 
