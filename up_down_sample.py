from scipy.signal import upfirdn
import numpy as np

def upsample(a:np.ndarray,p):

    assert a.ndim == 2
    return upfirdn([1],a,p,axis=1)

def downsample(a:np.ndarray,p):
    assert  a.ndim == 2
    return upfirdn([1],a,1,p,axis=1)

