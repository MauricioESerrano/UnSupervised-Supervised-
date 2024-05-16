import numpy as np
import matplotlib.pyplot as plt

# Using the Olivetti faces data set, which contains a collection of images of peoples’ faces.
# https://f000.backblazeb2.com/file/dsc-data/faces.csv

# PCA - Dimensional Reduction - 3 Images
# -----------------------------------------------
#                     A
# -----------------------------------------------

faces = np.loadtxt("faces.csv", delimiter=',')

C = np.cov(faces.T)

eigenvalues, eigenvectors = np.linalg.eigh(C)

u_1 = eigenvectors[:, -1]
u_2 = eigenvectors[:, -2]
u_3 = eigenvectors[:, -3]

u_1_reshape = u_1.reshape((64, 64))
u_2_reshape = u_2.reshape((64, 64))
u_3_reshape = u_3.reshape((64, 64))

plt.subplot(1, 3, 1)
plt.imshow(u_1_reshape, cmap='gray')

plt.subplot(1, 3, 2)
plt.imshow(u_2_reshape, cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(u_3_reshape, cmap='gray')

plt.show()

# PCA - dimensional Reduction - 18 Images
# -----------------------------------------------
#                     B
# -----------------------------------------------

faces = np.loadtxt("faces.csv", delimiter=',')

C = np.cov(faces.T)

eigenvalues, eigenvectors = np.linalg.eigh(C)

u_1 = eigenvectors[:, -1]
u_2 = eigenvectors[:, -2]
u_3 = eigenvectors[:, -3]

top1 = np.argsort(np.abs(faces @ u_1))[::-1]
top2 = np.argsort(np.abs(faces @ u_2))[::-1]
top3 = np.argsort(np.abs(faces @ u_3))[::-1]

bot1 = np.argsort(np.abs(faces @ u_1))[::1]
bot2 = np.argsort(np.abs(faces @ u_2))[::2]
bot3 = np.argsort(np.abs(faces @ u_3))[::1]

#Top 3
for i in range(3):

    plt.figure()

    plt.imshow(faces[top1[i]].reshape(64,-1))

for i in range(3):

    plt.figure()

    plt.imshow(faces[top2[i]].reshape(64,-1))

for i in range(3):

    plt.figure()

    plt.imshow(faces[top3[i]].reshape(64,-1))

# Bottom 3
for i in range(3):

    plt.figure()

    plt.imshow(faces[bot1[i]].reshape(64,-1))

for i in range(3):

    plt.figure()

    plt.imshow(faces[bot2[i]].reshape(64,-1))

for i in range(3):

    plt.figure()

    plt.imshow(faces[bot3[i]].reshape(64,-1))

plt.show()