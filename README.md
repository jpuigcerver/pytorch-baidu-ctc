# torch-baidu-ctc

[![Build Status](https://travis-ci.org/jpuigcerver/pytorch-baidu-ctc.svg?branch=master)](https://travis-ci.org/jpuigcerver/pytorch-baidu-ctc)

Pytorch bindings for Baidu's Warp-CTC. These bindings were inspired by
[SeanNaren's](https://github.com/SeanNaren/warp-ctc) but these include some bug fixes,
and offer some additional features.

```python
import torch
from torch_baidu_ctc import ctc_loss, CTCLoss

# Activations. Shape T x N x D.
# T -> max number of frames/timesteps
# N -> minibatch size
# D -> number of output labels (including the CTC blank)
x = torch.rand(10, 3, 6)
# Target labels
y = torch.tensor(
  [
    # 1st sample
    1, 1, 2, 5, 2,
    # 2nd
    1, 5, 2,
    # 3rd
    4, 4, 2, 3,
  ],
  dtype=torch.int,
)
# Activations lengths
xs = torch.tensor([10, 6, 9], dtype=torch.int)
# Target lengths
ys = torch.tensor([5, 3, 4], dtype=torch.int)

# By default, the costs (negative log-likelihood) of all samples are summed.
# This is equivalent to:
#   ctc_loss(x, y, xs, ys, average_frames=False, reduction="sum")
loss1 = ctc_loss(x, y, xs, ys)

# You can also average the cost of each sample among the number of frames.
# The averaged costs are then summed.
loss2 = ctc_loss(x, y, xs, ys, average_frames=True)

# Instead of summing the costs of each sample, you can perform
# other `reductions`: "none", "sum", or "mean"
#
# Return an array with the loss of each individual sample
losses = ctc_loss(x, y, xs, ys, reduction="none")
#
# Compute the mean of the individual losses
loss3 = ctc_loss(x, y, xs, ys, reduction="mean")
#
# First, normalize loss by number of frames, later average losses
loss4 = ctc_loss(x, y, xs, ys, average_frames=True, reduction="mean")


# Finally, there's also a nn.Module to use this loss.
ctc = CTCLoss(average_frames=True, reduction="mean", blank=0)
loss4_2 = ctc(x, y, xs, ys)

# Note: the `blank` option is also available for `ctc_loss`.
# By default it is 0.
```

## Requirements

- C++11 compiler (tested with GCC 4.9).
- Python: 2.7, 3.5, 3.6, 3.7 (tested with versions 2.7, 3.5 and 3.6).
- [PyTorch](http://pytorch.org/) >= 0.4.1 (tested with version 0.4.1).
- For GPU support: [CUDA Toolkit](https://developer.nvidia.com/cuda-zone).

## Installation

The installation process should be pretty straightforward assuming that you
have correctly installed the required libraries and tools.

The setup process compiles the package from source, and will compile with
CUDA support if this is available for PyTorch.

### From Pypi (recommended)

```bash
pip install torch-baidu-ctc
```

### From GitHub

```bash
git clone --recursive https://github.com/jpuigcerver/pytorch-baidu-ctc.git
cd pytorch-baidu-ctc
python setup.py build
python setup.py install
```

### AVX512 related issues

Some compiling problems may arise when using CUDA and newer host compilers
with AVX512 instructions. Please, install GCC 4.9 and use it as the host
compiler for NVCC. You can simply set the `CC` and `CXX` environment variables
before the build/install commands:

```bash
CC=gcc-4.9 CXX=g++-4.9 pip install torch-baidu-ctc
```

or (if you are using the GitHub source code):

```bash
CC=gcc-4.9 CXX=g++-4.9 python setup.py build
```

## Testing

You can test the library once installed using `unittest`. In particular,
run the following commands:

```bash
python -m unittest torch_baidu_ctc.test
```

All tests should pass (CUDA tests are only executed if supported).
