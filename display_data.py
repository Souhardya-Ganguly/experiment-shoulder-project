import numpy as np
import matplotlib.pyplot as plt

X = np.load("Preprocessed Data/data_train_b_full.npy")
Y = np.load("Preprocessed Data/data_mask_train_b_full.npy")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im0 = ax[0].imshow(X[0], cmap="gray")
ax[0].set_title("US Image")
im1 = ax[1].imshow(Y[0], cmap="gray")
ax[1].set_title("GT Mask")
for a in ax: a.axis("off")

plt.ion()
for i in range(X.shape[0]):
    im0.set_data(X[i])
    im1.set_data(Y[i])
    fig.suptitle(f"Index {i}")
    plt.pause(0.1)

plt.ioff()
plt.show()
