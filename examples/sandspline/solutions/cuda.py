import numpy as np
from numba import cuda
import numba
import time 

@cuda.jit
def splint(xa, ya, y2a, x, y):
    # s_xa = cuda.shared.array(100, dtype=numba.types.float64)
    # s_ya = cuda.shared.array((100, 2), dtype=numba.types.float64)
    # s_y2a = cuda.shared.array((100, 2), dtype=numba.types.float64)

    # ti = cuda.threadIdx.x
    # if ti < 100:
    #     s_xa[ti] = xa[ti]
    #     s_ya[ti,0] = ya[ti, 0]
    #     s_ya[ti,1] = ya[ti, 1]
    #     s_y2a[ti,0] = y2a[ti, 0]
    #     s_y2a[ti,1] = y2a[ti, 1]
    # cuda.syncthreads()

    i = cuda.grid(1)
    # klo = 0
    # khi = xa.shape[0]-1
    # while(khi-klo) > 1:
    #     k = (khi+klo) >> 1
    #     if s_xa[k] > x[i]:
    #         khi = k
    #     else:
    #         klo = k
    # h = s_xa[khi] - s_xa[klo]
    # a = (s_xa[khi]-x[i])/h    
    # b = (x[i]-s_xa[klo])/h
    # h = 1
    # a = 1
    # b = 1
    #y[i,0] = a*s_ya[klo, 0]+b*s_ya[khi, 0]+((a**3-a)*s_y2a[klo, 0]+(b**3-b)*s_y2a[khi, 0])*h**2/6.
    #y[i,1] = a*s_ya[klo, 1]+b*s_ya[khi, 1]+((a**3-a)*s_y2a[klo, 1]+(b**3-b)*s_y2a[khi, 1])*h**2/6.

theta = np.linspace(0, 2*np.pi, 100)
path = np.random.rand(100, 2)
y2 = np.random.rand(100, 2)

xs = np.linspace(0, 2.*np.pi, 1000)
ys = np.zeros((xs.size, 2))

d_xs = cuda.to_device(xs)
d_ys = cuda.to_device(ys)

d_theta = cuda.to_device(theta)
d_path = cuda.to_device(path)
d_y2 = cuda.to_device(y2)


threadsperblock = (16)
blockspergrid = xs.size // 128
splint[blockspergrid, threadsperblock](d_theta, d_path, d_y2, d_xs, d_ys)

t1 = time.time()
cuda.synchronize()
for i in range(100):
    #splint[blockspergrid, threadsperblock](d_theta, d_path, d_y2, d_xs, d_ys)
    splint[blockspergrid, threadsperblock](theta, path, y2, xs, ys)
    #cuda.synchronize()
cuda.synchronize()
t2 = time.time()

print(t2-t1)