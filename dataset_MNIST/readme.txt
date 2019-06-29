Native MNIST xdata format is 0 to 255, which is converted in these datasets to 0 to 1.

Exception: mnist_pca200: For this, since xdata is multiplied by PCA matrix, data range is approx -6 to 6. If desired, this can be minmax normalized to get range 0 to 1 for the NN. But this gives much worse results (around 5% test acc drop) as compared to the range -6 to 6.