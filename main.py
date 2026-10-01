import numpy as np
import struct

with open("data/train-images.idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)

images = data.reshape(num_images, rows * cols)
images = images / 255.0
mean = np.mean(images, axis=0)
centered = images - mean
cov = (centered.T @ centered) / (centered.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
