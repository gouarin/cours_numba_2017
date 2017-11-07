import numpy as np
import matplotlib.pyplot as plt
import os
from solutions.jit_gu import *
#from origin import *
import time

img_size = 5000
channels = 4
output_dir = './output/'

img = np.ones((img_size, img_size, channels), dtype=np.float32)
nspline = 75
theta = 2 * np.pi * np.linspace(0, 1, nspline)
r = 0.3
path = np.c_[r*np.cos(theta), r*np.sin(theta)]
guide = np.array([0.5,0.5])

nsamples = 1000
ys = np.zeros((nsamples, 2))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for j in range(5):
    t1 = time.time()
    for i in range(40000):
        y2 = spline(theta, path)
        xs = (np.random.random() + 2 * np.pi * np.linspace(0, 1, nsamples))%(2*np.pi)
        splint_2d(theta, path, y2, xs, ys)
        draw_pixel(img, ys[:, 0], ys[:, 1], guide)
        update_path(path, scale_value=.000005, periodic=True)
    t2 = time.time()
    print('Execution time:', t2-t1)

    ind = np.where(img>=1)
    img[ind] = 1
    ind = np.logical_and(img>0, img<1)
    img[ind] = img[ind]**2

    plt.imshow(img)
    plt.axes().set_aspect('equal')
    plt.axis('off')
    plt.savefig(output_dir + 'circle_{}_{}.png'.format(nspline, j), dpi=300, bbox_inches='tight')
