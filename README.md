# Pushing Boundaries even Further: Mixup's Influence on Neural Collapse

## Introduction

Neural collapse (NC; [[1]](#1)) is a phenomenon in deep learning that occurs when a neural network is trained
for classification and toward overfitting the training dataset. Under NC, the features of the layer before
the classification head exhibit the geometric structure of an Equiangular Tight Frame (ETF) and the decision
rule of the network collapses to a nearest-neighbor decision where the penultimate layer's features are
compared to the mean of the training examples' features.

Mixup [[3]](#3) is a data augmentation technique that works by creating a convex combination of two training
examples. In addition to providing a way of regularizing the model, it yields a more calibrated model.
Fisher et al. [[2]](#2) investigated how the success of Mixup could be explained by the prevalence of NC
in different stages of training and under different loss functions. Despite these efforts, it remains open
whether different loss functions exhibit different NC dynamics, as well as the impact of different network
architectures.

## Features

- Command-line interface for easy configuration and execution.
- Support for multiple datasets and model architectures.
- Configurable training parameters such as learning rate, weight decay, and number of epochs.
- Reproducibility through random seed setting.
- Visualization of model training progress.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train a model, run the following command:
```bash
python main.py --dataset cifar10 --model resnet18 --loss cross_entropy --epochs 500
```
where arguments such as `dataset`, `model`, `loss`, and `epochs` are variable. For a full list of
available arguments, refer to `main.py`.

To plot the last-layer representations and loss curves from a pkl file, run the following command:
```bash
python utils.py <command_type> ...
```

## References

<a id="1">[1]</a>
Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40), 24652-24663.

<a id="2">[2]</a>
Fisher, Q., Meng, H., & Papyan, V. (2024). Pushing Boundaries: Mixup's Influence on Neural Collapse. arXiv preprint arXiv:2402.06171.

<a id="3">[3]</a>
Zhang, H. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.
