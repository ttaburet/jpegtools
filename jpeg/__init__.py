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
        self.arr = arr
    
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
        return dct(dct(arr.T, norm = 'ortho').T, norm = 'ortho')

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
        return idct(idct(arr.T, norm = 'ortho').T, norm = 'ortho')

    def _blockwise2D(self, arr, block_size, f, shape = 'same'):
        
        h, w = arr.shape[:2]
    
        if(shape == 'vect'):
            res = np.zeros((h*w))
            area_block = block_size[0]*block_size[1]
            n_block_h, n_block_w = h//block_size[0], w//block_size[1]
            for i in range(n_block_h):
                for j in range(n_block_w):
                    res[area_block*i*(n_block_w)+area_block*j:area_block*i*n_block_w+area_block*(j+1)] = \
                        f(arr[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]]).flatten()
        else:
            res = np.zeros((h, w))

            for i in range(h//block_size[0]):
                for j in range(w//block_size[1]):
                    res[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] = \
                        f(arr[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]])

        return res
    
    @fluent
    def RGB2YCbCr(self, rounded = True):
        
        if(rounded):
            self.arr = self.arr.astype(np.uint8)
    
    @fluent
    def YCbCr2RGB(self, rounded = True):
        self.arr = np.dstack([np.array(0.299*self.arr[:, :, 0]+0.587*self.arr[:, :, 1]+0.114*self.arr[:, :, 2]),\
                              np.array(-0.1687*self.arr[:, :, 0]-0.3313*self.arr[:, :, 1]+0.5*self.arr[:, :, 2]+128),\
                              np.array(0.5*self.arr[:, :, 0]-0.4187*self.arr[:, :, 1]-0.0813*self.arr[:, :, 2]+128)])
        if(rounded):
            self.arr = self.arr.astype(np.uint8)
    
    @fluent
    def DCT(self, rounded = True):
        if(self.arr.shape[2] == 3):
            self.arr = np.dstack([self._blockwise2D(self.arr[:, :, i], (8, 8), self._dct2d) for i in range(3)]).astype(np.int)
        else:
            self.arr = self._blockwise2D(self.arr[:, :], (8, 8), self._dct2d).astype(np.int)
        
        if(rounded):
            self.arr = self.arr.astype(np.int)
    
    @fluent
    def IDCT(self, rounded = True):
        if(self.arr.shape[2] == 3):
            self.arr = np.dstack([self._blockwise2D(self.arr[:, :, i], (8, 8), self._idct2d) for i in range(3)])
        else:
            self.arr = self._blockwise2D(self.arr[:, :, i], (8, 8), self._idct2d)
        
        if(rounded):
            self.arr = self.arr.astype(np.int)
    
    @fluent
    def Quantize(self, Q, rounded = True):
        if(self.arr.shape[2] == 3):
            self.arr = np.dstack([self._blockwise2D(self.arr[:, :, i], (8, 8), lambda arr : np.multiply(arr, 1/Q[i])) for i in range(3)])
        else:
            self.arr = self._blockwise2D(self.arr, (8, 8), lambda arr : np.multiply(arr, 1/Q))
        
        if(rounded):
            self.arr = self.arr.astype(np.int)
    
    @fluent
    def Dequantize(self, Q, rounded = True):
        if(self.arr.shape[2] == 3):
            self.arr = np.dstack([self._blockwise2D(self.arr[:, :, i], (8, 8), lambda arr : np.multiply(arr, Q[i])) for i in range(3)])
        else:
            self.arr = self._blockwise2D(self.arr, (8, 8), lambda arr : np.multiply(arr, Q))
            
        if(rounded):
            self.arr = self.arr.astype(np.int)
    
    @fluent
    def Z_Ordering(self):
        if(self.arr.shape[2] == 3):
            self.arr = np.dstack([self._blockwise2D(self.arr[:, :, i], (8, 8), lambda arr : np.dot(arr.flatten(), M_P), 'vect') for i in range(3)])
        else:
            self.arr = self._blockwise2D(self.arr, (8, 8), lambda arr : np.dot(arr.flatten(), M_P), 'vect')
    
    @fluent
    def N_Ordering(self):
        if(self.arr.shape[2] == 3):
            self.arr = np.dstack([self._blockwise2D(self.arr[:, :, i], (8, 8), lambda arr : np.dot(arr.flatten(), M_P.T), 'vect') for i in range(3)])
        else:
            self.arr = self._blockwise2D(self.arr, (8, 8), lambda arr : np.dot(arr.flatten(), M_P.T), 'vect')
    
    def output(self):
        return self.arr