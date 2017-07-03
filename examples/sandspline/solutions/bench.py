import numpy as np
import time

def bench(module, guide):
    m = __import__(module)
    np.random.seed(42)
    size = 5000
    channels = 4
    img = np.ones((size, size, channels), dtype=np.float32)
    #n = np.random.randint(15, 100)
    n = 100
    theta = 2 * np.pi * np.linspace(0, 1, n)
    r = 0.3
    path = np.c_[r*np.cos(theta), r*np.sin(theta)]

    ys = np.zeros((1000, 2))

    bench = {}
    nrep = 1000

    y2 = m.spline(theta, path)
    xs = 2 * np.pi * np.linspace(0, 1, 1000)
    m.splint_2d(theta, path, y2, xs, ys)
    m.draw_pixel(img, ys[:, 0], ys[:, 1], guide)
    m.update_path(path, True)


    for i in range(nrep):
        t1 = time.time()
        y2 = m.spline(theta, path)
        t2 = time.time()
        bench['spline'] = bench.get('spline', 0) + t2-t1

        xs = (np.random.random() + 2 * np.pi * np.linspace(0, 1, 1000))%(2*np.pi)

        t1 = time.time()
        m.splint_2d(theta, path, y2, xs, ys)
        t2 = time.time()
        bench['splint'] = bench.get('splint', 0) + t2-t1

        t1 = time.time()
        m.draw_pixel(img, ys[:, 0], ys[:, 1], guide)
        t2 = time.time()
        bench['drawpixel'] = bench.get('drawpixel', 0) + t2-t1

        t1 = time.time()
        m.update_path(path, True)
        t2 = time.time()
        bench['update path'] = bench.get('update path', 0) + t2-t1

    print(bench)
    som = 0
    for v in bench.values():
        som += v
    print('total time', som)

modules = ['jit',
           'jit_gu']

for m in modules:
    guide = np.array([0.5,0.5])
    bench(m, guide)