import numpy as np

def load_data(filename):
    img = np.loadtxt(filename, delimiter=',', dtype=int)
    faces = img.reshape(400, 4096)
    return faces
