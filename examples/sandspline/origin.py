import numpy as np
import time

def spline(x, y):
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

def splint(xa, ya, y2a, x, y):
    n = xa.shape[0]
    for i in range(x.shape[0]):
        klo = 0
        khi = n-1
        while(khi-klo) > 1:
            k = (khi+klo) >> 1
            if xa[k] > x[i]:
                khi = k
            else:
                klo = k
        h = xa[khi] - xa[klo]
        a = (xa[khi]-x[i])/h    
        b = (x[i]-xa[klo])/h
        y[i,:] = a*ya[klo,:]+b*ya[khi,:]+((a**3-a)*y2a[klo,:]+(b**3-b)*y2a[khi,:])*h**2/6.

def draw_pixel(img, ys, guide):
    size = img.shape[0]
    coords = np.asarray(np.floor((guide + ys)*size), dtype='i8')
    color = np.array([0.0, 0.41568627450980394, 0.61960784313725492, 1.])*.0005
    pixels = img[coords[:, 0], coords[:, 1], :]
    invA = 1. - color[3]
    img[coords[:, 0], coords[:, 1], :] = color + pixels*invA

def update_path(path):
    np.random.seed(42)
    n = path.shape[0]
    scale = np.arange(n)*0.000003
    #r = 1.0-2.0*np.random.random(n)
    r = 1.0-2.0*np.linspace(0, 1, n)
    noise = r*scale
    print(noise)
    #phi = np.random.random(n)*2*np.pi
    phi = 2*np.pi*np.linspace(0, 1, n)
    rnd = np.c_[np.cos(phi), np.sin(phi)]
    path += rnd*noise[:, np.newaxis]
    path[-1] = path[0]

# np.random.seed(42)
# size = 5000
# channels = 4
# img = np.ones((size, size, channels), dtype=np.float32)
# #n = np.random.randint(15, 100)
# n = 100
# theta = 2 * np.pi * np.linspace(0, 1, n)
# r = 0.3
# path = np.c_[r*np.cos(theta), r*np.sin(theta)]
# guide = np.array([[0.5,0.5]])

# ys = np.zeros((1000, 2))

# bench = {}
# nrep = 100

# for i in range(nrep):
#     t1 = time.time()
#     y2 = spline(theta, path)
#     t2 = time.time()
#     bench['spline'] = bench.get('spline', 0) + t2-t1

#     xs = (np.random.random() + 2 * np.pi * np.linspace(0, 1, 1000))%(2*np.pi)

#     t1 = time.time()
#     splint(theta, path, y2, xs, ys)
#     t2 = time.time()
#     bench['splint'] = bench.get('splint', 0) + t2-t1

#     t1 = time.time()
#     draw_pixel(img, ys, guide)
#     t2 = time.time()
#     bench['drawpixel'] = bench.get('drawpixel', 0) + t2-t1

#     t1 = time.time()
#     update_path(path)
#     t2 = time.time()
#     bench['update path'] = bench.get('update path', 0) + t2-t1
    

# print(bench)
# som = 0
# for v in bench.values():
#     som += v
# print('total time', som)