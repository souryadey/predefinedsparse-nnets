# predefinedsparse-nnets

This repository implements **pre-defined sparse neural networks** -- as per research done by the [USC HAL team](https://hal.usc.edu/). Pre-defined sparsity lowers complexity of networks with minimal performance degradation. This leads to simpler architectures and better understanding of the 'black box' that is neural networks.

This **research paper** has more details. Please consider citing it if you use or benefit from this work:<br>
Sourya Dey, Kuan-Wen Huang, Peter A. Beerel, Keith M. Chugg, "Pre-Defined Sparse Neural Networks with Hardware Acceleration" in _IEEE Journal on Emerging and Selected Topics in Circuits and Systems_, vol. 9, no. 2, pp. 332-345, June 2019.<br>
Available on [IEEE](https://ieeexplore.ieee.org/document/8689061) and [arXiv](https://arxiv.org/abs/1812.01164) (copyright owned by IEEE).

<br>**Requirements**: Python 3, [Keras](https://keras.io/) (this work uses the Tensorflow backend), numpy, scipy

<br>**Main file**: [keras_impl](./keras_impl.py)
<br>Run the `sim_net` method with these arguments:
- `config`: Neuron configuration
- `fo`: Out-degree (fanout) configuration
- `l2_val`: L2 regularization coefficient
- `z`: Degree of parallelism, if simulating clash-free adjacency matrices
- `dataset_filename`: Path to dataset
- `preds_compare`: Path to benchmark results with which current results will be compared. Some benchmark results are in [timit_FCs](./timit_FCs/)

For example:
```
recs,model = sim_net(
                    config = np.array([800,100,10]),
                    fo = np.array([50,10]),
                    l2_val = 8e-5,
                    z = None,
                    dataset_filename = data_folder + 'dataset_MNIST/mnist.npz',
                    preds_compare = 0
                    )
```

A complete explanation of these terms and concepts is given in the research paper. The documentation of the `run_model` method also has useful details. Final results and the model itself after doing a run are stored in [results_new](./results_new/) by default (some examples are given).

<br>**Supporting files**:
- [adjmatint](./adjmatint.py): Create adjacency matrices which describe the pre-defined sparse connection pattern in different junctions. Different types -- random, basic, clash-free.
- [data_loadstore](./data_loadstore.py): Data management (see datasets section below).
- [data_processing](./data_processing.py): Normalization functions.
- [keras_nets](keras_nets.py): Methods to create MLPs, as well as conv nets used in the research paper to experiment on CIFAR.
- [utils](./utils.py): Has the useful `merge_dicts` method for managing training records.

<br>**Datasets for experimentation** are used in the .npz format with 6 keys -- `xtr, ytr, xva, yva, xte, yte` for data (x) and labels (y) of training, validation and test splits. Our experiments included:
- [MNIST](./dataset_MNIST/)
- [CIFAR](./dataset_CIFAR/)
- [Reuters RCV1 v2](./dataset_RCV1/) - Links to download and methods to process this dataset are given in [data_loadstore](./data_loadstore.py)
- [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1): Not freely available, hence not provided.

<br>**Further research details**: Our group at USC has been reseaching and developing pre-defined sparsity starting from 2016. Our other publications can be found [here](https://hal.usc.edu/publications.html). An associated effort - the development of synthetic datasets for classifying Morse code symbols - can be found [here](https://github.com/usc-hal/morse-dataset).
