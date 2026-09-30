import numpy as np
import struct

with open("data/train-images.idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)

images = data.reshape(num_images, rows * cols)

print(magic)
print(num_images)
print(rows, cols)
print(images.shape)
print(images.dtype)