import numpy as np
import struct
import matplotlib.pyplot as plt

with open("data/train-images.idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)

images = data.reshape(num_images, rows * cols)
images = images / 255.0
mean = np.mean(images, axis=0)
centered = images - mean
cov = (centered.T @ centered) / (centered.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]
eigenvalues = eigenvalues[:30]
eigenvectors = eigenvectors[:, :30]

""""
# Used to plot the top 30 eigenvalues
plt.figure(figsize=(8, 5))
plt.plot(range(1, 31), eigenvalues, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue (Variance)")
plt.title("Top 30 eigenvalues")
plt.grid(True, alpha=0.3)
plt.show()
"""

"""
# Used to plot the top 30 eigenvectors
fig, axes = plt.subplots(5, 6, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenvectors[:, i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()
"""

