import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from origin import *
import time

GRID_Y = 12

lo, up = .2, .8
path = []
for i, y in enumerate(np.linspace(lo, up, GRID_Y)):
    pnum = 4+i
    ## hlines
    x = np.linspace(lo, up, pnum)
    yy = y*np.ones(pnum)
    path.append(np.c_[x, yy])

img_size = 5000
channels = 4
output_dir = './output/'

img = np.ones((img_size, img_size, channels), dtype=np.float32)

nsamples = 500
ys = np.zeros(nsamples)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for j in range(10):
    t1 = time.time()
    for i in range(40000):
        for ip, p in enumerate(path):
            y2 = spline(p[:, 0], p[:, 1])
        
            xs = .001*np.random.random() + np.linspace(lo+.05,up-.05, nsamples)
        
            splint_1d(p[:, 0], p[:, 1], y2, xs, ys)
            draw_pixel(img, ys, xs, np.zeros(2))
            update_path(path[ip])

    t2 = time.time()
    print('Execution time:', t2-t1)

    ind = np.where(img<=0)
    img[ind] = 0
    ind = np.where(img>=1)
    img[ind] = 1
    ind = np.logical_and(img>0, img<1)
    img[ind] = img[ind]**2

    plt.imshow(img, origin='lower')
    plt.axes().set_aspect('equal')
    plt.axis('off')
    plt.savefig(output_dir + 'hlines_{}.png'.format(j), dpi=300, bbox_inches='tight')