import numpy as np
from numba import njit, guvectorize, generated_jit, types
import math

__all__ = ['spline', 'splint_1d', 'splint_2d', 'draw_pixel', 'update_path']

@njit
def spline_1d(x, y):
    n = x.shape[0]
    u = np.zeros_like(y)
    y2 = np.zeros_like(y)

    #dif = np.diff(x)
    for i in range(1, n-1):
        x1 = x[i]-x[i-1]
        x2 = x[i+1]-x[i]
        sig = x1/(x[i+1]-x[i-1])
        p = sig*y2[i-1] + 2.
        y2[i] = (sig-1)/p
        u[i] = (y[i+1]- y[i])/x2 - (y[i]-y[i-1])/x1
        u[i] = (6*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p
    
    for i in range(n-2, -1, -1):
        y2[i] = y2[i]*y2[i+1]+u[i]

    return y2

@njit
def spline_2d(x, y):
    n = x.shape[0]
    u = np.zeros_like(y)
    y2 = np.zeros_like(y)

    dif = np.diff(x)
    for i in range(1, n-1):
        sig = dif[i-1]/(x[i+1]-x[i-1])
        for k in range(y.shape[1]):
            p = sig*y2[i-1, k] + 2.
            y2[i, k] = (sig-1)/p
            u[i, k] = (y[i+1, k]- y[i, k])/dif[i] - (y[i, k]-y[i-1, k])/dif[i-1]
            u[i, k] = (6*u[i, k]/(x[i+1]-x[i-1])-sig*u[i-1, k])/p
    
    for i in range(n-2, -1, -1):
        for k in range(y.shape[1]):
            y2[i, k] = y2[i, k]*y2[i+1, k]+u[i, k]

    return y2

@generated_jit(nopython=True)
def spline(x, y):
    if isinstance(y, types.Array) and y.ndim == 1:
        return spline_1d
    elif isinstance(y, types.Array) and y.ndim == 2:
        return spline_2d

@njit
def draw_pixel(img, xs, ys, guide):
    size = img.shape[0]
    color = np.array([0.0, 0.41568627450980394, 0.61960784313725492, 1.])*.0005
    invA = 1. - color[3]
    for i in range(ys.shape[0]):
        x = math.floor((guide[0] + xs[i])*size)
        y = math.floor((guide[1] + ys[i])*size)
        if 0 <= x < size and 0 <= y < size:
            for k in range(4):
                img[x, y, k] = color[k] + img[x, y, k]*invA

@njit
def update_path(path, periodic=False, scale_value=0.00001):
    n = path.shape[0]

    for i in range(n):
        scale = i*scale_value
        r = 1.0-2.0*np.random.random()
        noise = r*scale
        phi = np.random.random()*2*np.pi
        path[i, 0] += np.cos(phi)*noise
        path[i, 1] += np.sin(phi)*noise
        
    if periodic:
        path[n-1, 0] = path[0, 0]
        path[n-1, 1] = path[0, 1]

@guvectorize(['void(float64[:], float64[:], float64[:], float64[:], float64[:])'], "(n),(n),(n),() -> ()", target="parallel", nopython=True)    
def splint_1d(xa, ya, y2a, x, y):
    klo = 0
    khi = xa.shape[0]-1
    while(khi-klo) > 1:
        k = (khi+klo) >> 1
        if xa[k] > x[0]:
            khi = k
        else:
            klo = k
    h = xa[khi] - xa[klo]
    a = (xa[khi]-x[0])/h    
    b = (x[0]-xa[klo])/h
    y[0] = a*ya[klo]+b*ya[khi]+((a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*h**2/6.

@guvectorize(['void(float64[:], float64[:,:], float64[:,:], float64[:], float64[:])'], "(n),(n, n2),(n, n2),() -> (n2)", target="parallel", nopython=True)    
def splint_2d(xa, ya, y2a, x, y):
    klo = 0
    khi = xa.shape[0]-1
    while(khi-klo) > 1:
        k = (khi+klo) >> 1
        if xa[k] > x[0]:
            khi = k
        else:
            klo = k
    h = xa[khi] - xa[klo]
    a = (xa[khi]-x[0])/h    
    b = (x[0]-xa[klo])/h
    for k in range(y.shape[0]):
        y[k] = a*ya[klo, k]+b*ya[khi, k]+((a**3-a)*y2a[klo, k]+(b**3-b)*y2a[khi, k])*h**2/6.