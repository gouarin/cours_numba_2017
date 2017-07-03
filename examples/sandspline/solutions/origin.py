import numpy as np
import time

def spline_1d(x, y):
    n = x.shape[0]
    u = np.zeros_like(y)
    y2 = np.zeros_like(y)

    dif = np.diff(x)
    sig = dif[:-1]/(x[2:]-x[:-2])

    u[1:-1] = (y[2:]- y[1:-1])/dif[1:] - (y[1:-1]-y[:-2])/dif[:-1]

    for i in range(1, n-1):
        p = sig[i-1]*y2[i-1] + 2.
        y2[i] = (sig[i-1]-1)/p
        u[i] = (6*u[i]/(x[i+1]-x[i-1])-sig[i-1]*u[i-1])/p
    
    for i in range(n-2, -1, -1):
        y2[i] = y2[i]*y2[i+1]+u[i]

    return y2

def spline_2d(x, y):
    n = x.shape[0]
    u = np.zeros_like(y)
    y2 = np.zeros_like(y)

    dif = np.diff(x)
    sig = dif[:-1]/(x[2:]-x[:-2])

    u[1:-1, :] = (y[2:,:]- y[1:-1,:])/dif[1:, np.newaxis] - (y[1:-1,:]-y[:-2,:])/dif[:-1, np.newaxis]

    for i in range(1, n-1):
        p = sig[i-1]*y2[i-1,:] + 2.
        y2[i, :] = (sig[i-1]-1)/p
        u[i, :] = (6*u[i,:]/(x[i+1]-x[i-1])-sig[i-1]*u[i-1,:])/p
    
    for i in range(n-2, -1, -1):
        y2[i,:] = y2[i,:]*y2[i+1,:]+u[i,:]

    return y2

def spline(x, y):
    if y.ndim == 2:
	    return spline_2d(x, y)
    else:
	    return spline_1d(x, y)

def splint_1d(xa, ya, y2a, x, y):
    khi = np.searchsorted(xa, x)
    klo = khi-1
    h = xa[khi] - xa[klo]
    a = (xa[khi]-x)/h
    b = (x-xa[klo])/h
    y[:] = a*ya[klo]+b*ya[khi]+((a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*h**2/6.

def splint_2d(xa, ya, y2a, x, y):
    khi = np.searchsorted(xa, x)
    klo = khi-1
    h = xa[khi] - xa[klo]
    a = ((xa[khi]-x)/h)[:, np.newaxis]
    b = ((x-xa[klo])/h)[:, np.newaxis]
    h = h[:, np.newaxis]
    y[:] = a*ya[klo]+b*ya[khi]+((a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*h**2/6.

def draw_pixel(img, xs, ys, guide):
    size = img.shape[0]
    newxs = np.floor((guide[0] + xs)*size)
    xs_mask = np.logical_and(newxs>=0, newxs<size)
    newys = np.floor((guide[1] + ys)*size)
    ys_mask = np.logical_and(newys>=0, newys<size)
    mask = np.logical_and(xs_mask, ys_mask)
    coords = np.asarray([newxs[mask],newys[mask]], dtype='i8')
    color = np.array([0.0, 0.41568627450980394, 0.61960784313725492, 1.])*.0005
    pixels = img[coords[0, :], coords[1, :], :]
    invA = 1. - color[3]
    img[coords[0, :], coords[1, :], :] = color + pixels*invA

def update_path(path, periodic=False, scale_value=0.00001):
    n = path.shape[0]
    scale = np.arange(n)*scale_value
    r = 1.0-2.0*np.random.random(n)
    noise = r*scale
    phi = np.random.random(n)*2*np.pi
    rnd = np.c_[np.cos(phi), np.sin(phi)]
    path += rnd*noise[:, np.newaxis]
    if periodic:
        path[-1] = path[0]
