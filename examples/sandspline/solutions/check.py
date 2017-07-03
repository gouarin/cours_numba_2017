import numpy as np
from origin import spline, splint, update_path

def check_spline(module):
    n = 10
    theta = 2 * np.pi * np.linspace(0, 1, n)
    r = 0.3
    path = np.c_[r*np.cos(theta), r*np.sin(theta)]

    ref = spline(theta, path)

    m = __import__(module)
    out = m.spline(theta, path)
    assert(np.allclose(ref, out))

def check_splint(module):
    n = 10
    xa = np.linspace(0, 1, n)
    ya = np.random.rand(n, 2)
    y2 = np.random.rand(n, 2)

    nsamples = 100
    x = np.linspace(0.01, 9.99, nsamples)
    ref = np.zeros((nsamples, 2))
    out = np.zeros((nsamples, 2))

    splint(xa, ya, y2, x, ref)

    m = __import__(module)
    m.splint(xa, ya, y2, x, out)
    assert(np.allclose(ref, out)) 

modules = ['jit',
           'jit_gu']

for m in modules:
    check_spline(m)
    check_splint(m)