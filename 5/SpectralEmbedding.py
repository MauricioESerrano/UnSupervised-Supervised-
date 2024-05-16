import pandas as pd
import numpy as np
import sklearn.neighbors
import matplotlib.pyplot as plt

# Create plot using "Animals with Attributes Dataset, v1.0, May 13th 2009" and see similarities.
# Method is Spectral Embedding - K-neighbors 


class_names = pd.read_csv('classes.txt', sep='\t', header=None)
attribute_names = pd.read_csv('predicates.txt', sep='\t', header=None)
data = pd.read_fwf('predicate-matrix-continuous.txt', header=None)
data = data.rename(columns=attribute_names.iloc[:, 1])
data = data.set_index(class_names.iloc[:, 1].rename('class_name'))

A = data.to_numpy()

X = sklearn.neighbors.kneighbors_graph(A, n_neighbors = 10).toarray()

W = np.maximum(X, X.T)

D = np.diag(W.sum(axis=0))

L = D - W

values, vectors = np.linalg.eigh(L)

greaterThan = np.where(values > 1e-10)[0][0]

embedding = vectors[:, greaterThan:greaterThan+2]

figure, axes = plt.subplots()
figure.set_size_inches((10,8))
figure.tight_layout()
axes.scatter(*embedding.T)

for i, (x,y) in enumerate(embedding):
    axes.annotate(data.index[i], (x,y))

plt.show()


