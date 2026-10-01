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

# Used to plot the top 30 eigenvalues
""""
plt.figure(figsize=(8, 5))
plt.plot(range(1, 31), eigenvalues, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue (Variance)")
plt.title("Top 30 eigenvalues")
plt.grid(True, alpha=0.3)
plt.show()
"""

# Used to plot the top 30 eigenvectors

fig, axes = plt.subplots(5, 6, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenvectors[:, i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Eigenvector {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()


with open("data/train-labels.idx1-ubyte", "rb") as f:
    magic, num_labels = struct.unpack(">II", f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)

# Used to plot the different digits with varying PCAs and calculate the MSE
"""

digit_indices = [1,3,5,7,2,0,13,15,17,4]


ks = [2, 5, 20, 30]
mse_table = np.zeros((10, len(ks)))

def reconstruct(x, k):
    V = eigenvectors[:, :k]
    return ((x - mean) @ V) @ V.T + mean

for d, idx in enumerate(digit_indices):
    x = images[idx]
    for j, k in enumerate(ks):
        recon = reconstruct(x, k)
        mse_table[d, j] = np.mean((x - recon) ** 2)

fig, axes = plt.subplots(10, len(ks) + 1, figsize=(10, 16))

for d, idx in enumerate(digit_indices):
    x = images[idx]
    axes[d, 0].imshow(x.reshape(28, 28), cmap="gray")
    axes[d, 0].set_title(f"digit {d}", fontsize=8)
    for j, k in enumerate(ks):
        axes[d, j + 1].imshow(reconstruct(x, k).reshape(28, 28), cmap="gray")
        axes[d, j + 1].set_title(f"k={k}\nMSE={mse_table[d, j]:.4f}", fontsize=8)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
"""