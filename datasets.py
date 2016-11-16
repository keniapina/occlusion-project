import numpy as np

def gen_bars(n,dim=4, bw=False):
    """
    creates images where colored bars are ordered and stacked
    
    Parameters
    ----------
    n : int
        number of images
    dim : int
        dimension 
    
    Returns
    -------
    im : array
        image of bars
    barcolors : array
        colors 
    """
    if bw:
        barcolors=np.ones(2*dim)
        im=np.zeros((n,dim,dim))
    else:
        barcolors=np.random.rand(2*dim,3)
        im=np.zeros((n,dim,dim,3))
    j=0
    while j<n:
        x=np.random.binomial(1,1./dim,size=2*dim)
        if x.sum()>0:
            idx=np.nonzero(x)[0]
            idxp=np.random.permutation(idx)
            for i in idxp:
                if i>(dim-1):
                    if bw:
                        im[j,:,i-dim]=1.
                    else:
                        im[j,:,i-dim]=barcolors[i]
                else:
                    if bw:
                        im[j,i,:]=1.
                    else:
                        im[j,i,:]=barcolors[i]
            j+=1
    return im,barcolors

