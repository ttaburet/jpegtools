import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import functools
from scipy.fftpack import dct, idct

M_P = np.load('jpeg/JPEG_permutation.npy').astype(int)

def fluent(func):
    """
        Decorator for Fluent interface :
        classname().method1().method2().method3()
        Avoiding using 'return self' at the end of method1, method2, method3
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        self = args[0]
        func(*args, **kwargs)
        return self
    return wrapped

class JPEG():
    
    def __init__(self, arr):
        try:
            c = arr.shape[2]
        except IndexError:
            c = 1
        finally:
            temp = arr.copy()
            self.arr = arr.copy().reshape(temp.shape[0], temp.shape[1], c)
    
    def _dct2d(self, arr):
        """
        2D type-II discrete cosine transform (DCT)

        Parameters
        ----------
        arr : ndarray int
            Spatial array

        Returns
        -------
        ndarray float

            DCT-2D
        """

        return dct(dct(arr-128, norm='ortho', axis=2), norm='ortho', axis=3)

    def _idct2d(self, arr):
        """
        2D type-II inverse discrete cosine transform (IDCT)

        Parameters
        ----------
        arr : ndarray int
            DCT array

        Returns
        -------
        ndarray float

            Spatial array
        """

        return idct(idct(arr, norm='ortho', axis=2), norm='ortho', axis=3) + 128
    
    @fluent
    def RGB2YCbCr(self, rounded = True):
        self.arr = np.dstack([np.array(0.299*self.arr[:, :, 0]+0.587*self.arr[:, :, 1]+0.114*self.arr[:, :, 2]),\
                              np.array(-0.1687*self.arr[:, :, 0]-0.3313*self.arr[:, :, 1]+0.5*self.arr[:, :, 2]+128),\
                              np.array(0.5*self.arr[:, :, 0]-0.4187*self.arr[:, :, 1]-0.0813*self.arr[:, :, 2]+128)])
        # if(rounded):
        #     self.arr = self.arr.astype(np.uint8)
    
    @fluent
    def YCbCr2RGB(self, rounded = True):
        self.arr = np.dstack([np.array(0.299*self.arr[:, :, 0]+0.587*self.arr[:, :, 1]+0.114*self.arr[:, :, 2]),\
                              np.array(-0.1687*self.arr[:, :, 0]-0.3313*self.arr[:, :, 1]+0.5*self.arr[:, :, 2]+128),\
                              np.array(0.5*self.arr[:, :, 0]-0.4187*self.arr[:, :, 1]-0.0813*self.arr[:, :, 2]+128)])
        # if(rounded):
        #     self.arr = self.arr.astype(np.uint8)

    @fluent
    def DCT(self):
        self.arr = np.stack([self._dct2d(self.arr[:,:,:,:, i]) for i in range(self.arr.shape[-1])], axis = -1)
    
    @fluent
    def IDCT(self):
        self.arr = np.stack([self._idct2d(self.arr[:,:,:,:, i]) for i in range(self.arr.shape[-1])], axis = -1)
    
    @fluent
    def Round(self, func = None):
        if(func == None):
            self.arr = np.round(self.arr).astype(int)
        else:
            self.arr = func(self.arr).astype(int)

    @fluent
    def Quantize(self, Q, rounded = True):
        self.arr = np.stack([np.multiply(self.arr[:,:,:,:, i], 1.0/Q[i]) for i in range(self.arr.shape[-1])], axis = -1)

        # if(rounded):
        #     self.arr = np.round(self.arr).astype(int)
    
    @fluent
    def Dequantize(self, Q, rounded = True):
        self.arr = np.stack([np.multiply(self.arr[:,:,:,:, i], Q[i]) for i in range(self.arr.shape[-1])], axis = -1)
    
        # if(rounded):
        #     self.arr = self.arr.astype(np.int)
    
    @fluent
    def Z_Ordering(self):
        ravel = lambda arr : arr.reshape((arr.shape[0], arr.shape[1], 64))
        ravelBack = lambda arr : arr.reshape((arr.shape[0], arr.shape[1], 8, 8))
        self.arr = np.stack([ ravelBack(np.dot(ravel(self.arr[:,:,:,:, i]), M_P)) for i in range(self.arr.shape[-1])], axis = -1)
                
    @fluent
    def N_Ordering(self):
        ravel = lambda arr : arr.reshape((arr.shape[0], arr.shape[1], 64))
        ravelBack = lambda arr : arr.reshape((arr.shape[0], arr.shape[1], 8, 8))
        self.arr = np.stack([ ravelBack(np.dot(ravel(self.arr[:,:,:,:, i]), M_P.T)) for i in range(self.arr.shape[-1])], axis = -1)    
    
    @fluent
    def toBlocksView(self):
        h, w, c = self.arr.shape[:3]
        shape = (h//8, w//8, 8, 8)

        strided = np.zeros(list((h//8, w//8, 8, 8))+[c], dtype=self.arr.dtype)
        for i in range(c):
            sz = self.arr[:, :, i].itemsize
            strides = sz*np.array([w*8,8,w,1])

            blocks = np.lib.stride_tricks.as_strided(self.arr[:, :, i].copy(),\
                                            shape=shape,\
                                            subok=True,\
                                            strides=strides,\
                                            writeable = False)
            strided[:, :, :, :, i] = blocks

        self.arr = strided

    @fluent
    def toStandardView(self):
        self.arr = np.stack([self.arr[:,:,:,:,i].swapaxes(1,2).reshape(self.arr.shape[0]*8, self.arr.shape[1]*8) for i in range(self.arr.shape[-1])], axis = -1)

    def output(self):
        return self.arr