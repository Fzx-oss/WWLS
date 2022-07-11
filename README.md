# Wasserstein Graph Distance based on L1-Approximated Tree Edit Distance between Weisfeiler–Lehman Subtrees
## Wasserstein Weisfeiler-Lehman Subtree (WWLS) Distance/Kernel

The WWLS is a graph metric/kernel that can measure the distance/similarity between graphs. As the name implies, this is based on the Wasserstein distance and WL test. Please see the paper for details. 

Our code uses PyTorch >= 1.10 and PyG for TUD datasets and C++14 for the tree calculation, so please install PyTorch >= 1.10 and the corresponding PyG environment on your machine and build a C++ environment beforehand.

[PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Setup

```shell
# Install the Python module and compile the C++ code as a Python module.
sudo bash setup.sh
```

## Usage

### Example:
```python
cd quick_start

# Quick tart (graph classification experiments)
python main.py

# Run PTC-MR with 2 iterations
cd quick_start
python main.py --dataset PTC_MR --h 2
```

### Optional arguments:
We provide the following optional arguments in ./quick_start/main.py

|  Parameter |  Defalut  | Description |
| ---- | ---- | ---- |
|  --dataset  | MUTAG | dataset name |
|  --h  |  2 | max iteration number |
|  --mode  |  s  | s: SVM, k:k-NN  |
| --C | 10  |  parameter C of SVM |
| --gamma | 0.1 | parameter gamma of graph kernel  |
| --k | 1 | parameter k of k-NN |
| --w | 1  | write cost matrices |

### Other experiments:
We also have code for the second and third experiments in the paper in the ''examples'' directory. You could use Jupyter notebook/lab to run them.

```python
# Experiments to measure maximum performance of graph classification
examples/maximum_performace.ipynb

# Experiments to verify performance as a metric
examples/distance.ipynb
```


## License
[MIT](https://choosealicense.com/licenses/mit/)