import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_hastie_10_2

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(421)
plt.title("One informative feature, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

plt.subplot(422)
plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

plt.subplot(423)
plt.title("Two informative features, two clusters per class", fontsize='small')
X2, Y2 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2)
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)

plt.subplot(424)
plt.title("Multi-class, two informative features, one cluster",
          fontsize='small')
X1, Y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

plt.subplot(425)
plt.title("Three blobs", fontsize='small')
X1, Y1 = make_blobs(n_samples=1000, n_features=2, centers=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

plt.subplot(426)
plt.title("Gaussian divided into four quantiles", fontsize='small')
X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=4)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

plt.subplot(427)
plt.title("hastie data ", fontsize='small')
X1, Y1 = make_hastie_10_2(n_samples=1000)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()