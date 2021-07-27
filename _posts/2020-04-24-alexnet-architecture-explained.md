---
title: "AlexNet Architecture Explained"
date: "2020-04-24"
categories: 
  - "computer-vision"
  - "deep-learning"
coverImage: "AlexNet-1.png"
---

AlexNet famously won the 2012 ImageNet LSVRC-2012 competition by a large margin (15.3% vs 26.2%(second place) error rates). Here is the link to original [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

Major highlights of the paper

1. Used ReLU instead of tanh to add non-linearity.
2. Used dropout instead of regularization to deal with overfitting.
3. Overlap pooling was used to reduce the size of the network.

**1\. Input**

AlexNet solves the problem of image classification with subset of ImageNet dataset with roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images. The input is an image of one of 1000 different classes and output is a vector of 1000 numbers.

The input to AlexNet is an RGB image of size 256\*256. This mean that all the images in training set and test images are of size 256\*256. If the input image is not 256\*256, image is rescaled such that shorter size is of length 256, and cropped out the central 256\*256 patch from the resulting image.

![](https://prabinnepal.com/wp-content/uploads/2020/09/AlexNet-Resize-Crop-Input.jpg)

[source](https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-Resize-Crop-Input.jpg)

The image is trained with raw RGB values of pixels. So, if input image is grayscale, it is converted into RGB image . Images of size 257\*257 were generated from 256\*256 images through random crops and it is feed to the first layer of AlexNet.

**2\. AlexNet Architecture**

AlexNet contains five convolutional layers and three fully connected layers - total of eight layers. AlexNet architecture is shown below:

![](https://prabinnepal.com/wp-content/uploads/2020/09/AlexNet-1.png)

AlexNet Architecture

[source](https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png)

For the first two convolutional layers, each convolutional layers is followed by a Overlapping Max Pooling layer. Third, fourth and fifth convolution layers are directly connected with each other. The fifth convolutional layer is followed by Overlapping Max Pooling Layer, which is then connected to fully connected layers. The fully connected layers have 4096 neurons each and the second fully connected layer is feed into a softmax classifier having 1000 classes. 

_2.1) ReLU Non-Linearity:_

The standard way of introducing nonlinearity is using tanh: f(x) = tanh(x) where f is a function of input x or using f(x) = (1+e^-x)^-1. 

These are saturating nonlinearities which are much slow than non-saturating nonlinearity f(x) = max(0, x), in terms of training time with gradient descent.

![](https://prabinnepal.com/wp-content/uploads/2020/09/activation-functions.png)

fig. (Tanh and Relu activation functions)

Saturating nonlinearities**:** These functions have a compact range, meaning that they compress the neural response into a bounded subset of the real numbers. The LOG compresses inputs to outputs between 0 and 1, the TAN H between -1 and 1. These functions display limiting behavior at the boundaries.

Training network with non-saturating nonlinearity is faster than that of saturating non-linearity.

_2.2) Overlapping Pooling:_

Max Pooling layers help to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned. Max Pooling helps to reduce overfitting. Basically, it uses a max operation to pool sets of features, leaving us with a smaller number of them. Max Pooling and Overlapping is same except except the adjacent windows over which the max is computed overlap each other.

> "A pooling layer can be thought of as consisting of a grid of pooling units spaced s pixels apart, each summarizing a neighbourhood of size z\*z centered at the location of pooling unit. If we set s=z, we obtain traditional local pooling. If we set s < z, we obtain overlapping pooling.
> 
> AlexNet Paper (2012)

The overlapping pooling reduces the top-1 and top-5 error rates by 0.4% and 0.3% compared to non-overlapping pooling, thus finding it very difficult to overfit.

2.3) Reducing Overfitting

Various techniques are applied to reduce overlapping

**Data Augmentation**

The most common way to reduce overfitting on image data is data augmentation. It is a strategy to significantly increase the diversity of data available for training the models without collecting new data. Data augmentation includes techniques such as Position augmentation (cropping, padding, rotating, translation, affine transformation), color augmentation(Brightness, contrast saturation, hue) and many other. AlexNet employ two distinct forms of data augmentation. 

The first form of data augmentation is translating the image and horizontal reflections. This is done by extracting random 224\*224 patches from 256\*256 images and training network on these patches. The second form of data augmentation consists of altering the intensities of RGB channel in training images.

**Dropout**

Dropout is a regularization technique to reduce overfitting and improving generalization of deep neural networks. 'Dropout' refers to dropping out units(hidden and visible) in a neural network. We can interpret dropout as the probability of training  a given node in a layer, where 1.0 means no dropout and 0.5 means 50% of hidden neurons are ignored.

![](https://prabinnepal.com/wp-content/uploads/2020/09/dropout.png)

Dropout

[source](http://jmlr.org/papers/v15/srivastava14a.html)

References:

1. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, 2012
2. [https://www.learnopencv.com/understanding-alexnet/](https://www.learnopencv.com/understanding-alexnet/)
