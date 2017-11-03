
# Text Detection FCN
A Caffe Fully Convolutional Network that detects any kind of text and generates pixel-level heatmaps. 
Adapted from [Fully Convolutional Models for Semantic Segmentation](https://github.com/shelhamer/fcn.berkeleyvision.org) by Jonathan Long, Evan Shelhamer, and Trevor Darrell. CVPR 2015 and [PAMI 2016](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

## Model
A COCO-Text trained model is [available here](https://drive.google.com/file/d/0B-DM8FPBNpG6QVRLSFBNQ1dLalU/view?usp=sharing).

## Publications
This FCN was used to improve the [TextProposals](https://github.com/lluisgomez/TextProposals) algorithm by Lluis Gomez. The improved version is [available here](https://github.com/gombru/TextProposalsInitialSuppression). That lead to two publications, which you may cite if using this FCN:

[FAST: Facilitated and Accurate Scene Text Proposals through FCN Guided Pruning](http://www.sciencedirect.com/science/article/pii/S0167865517302982)
Dena Bazazian, Raul Gomez, Anguelos Nicolaou, Lluis Gomez, Dimosthenis Karatzas, Andrew D.Bagdanov. Pattern Recognition Letters. 2017.

[Improving Text Proposals for Scene Images with Fully Convolutional Networks](https://arxiv.org/abs/1702.05089)
Dena Bazazian, Raul Gomez, Anguelos Nicolaou, Lluis Gomez, Dimosthenis Karatzas and Andrew Bagdanov. DLPR2016. 2016.

## Demo**
The FCN can run in real time in a GPU

![](fcn_demo.gif)

## Requirements
The amount of required GPU memory depends on the image size:
1000x1000 --> 2.8 GB
512x512 --> 1.6 GB
It takes 0.17s per image on a TitanX.


## Development
This FCN was trained during MS’s thesis. Extra information about the training and usage can be found in the [thesis](https://drive.google.com/file/d/0B-DM8FPBNpG6QXdQN3JaY3pBMFU/view) or in this [slides](https://docs.google.com/presentation/d/1mu7wdI4DUGxHuF_bshniV8mMWiXrQmoB18nfcIBU-as/edit?usp=sharing).
