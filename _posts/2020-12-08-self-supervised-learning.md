---
title: "Self-supervised Learning"
date: "2020-12-08"
categories: 
  - "deep-learning"
  - "self-supervised-learning"
coverImage: "self-supervised-learning.png"
---

I have been exploring self-supervised learning and been through papers and blogs to understand it. Self-supervised learning is considered the next big thing in deep learning and why not! If there is a way to learn without providing labels, then this enables us to leverage a large amount of unlabeled data for our tasks. I am going to provide my understanding of self-supervised learning and will try to explain some papers about it.

We have been familiar with supervised learning wherein we provide features and labels to train a model and the model uses those labels to learn from the features. But labeling data is not an easy task as it requires more time and manpower. There is a large amount of data being generated daily and is unlabeled. The generated data may be in the form of text, images, audio, or videos. Those data can be used for different purposes. But there is a catch. These data do not contain labels and it difficult to work on these sorts of data. Here comes self-supervised learning to the rescue.

### So what is self-supervised learning and why is it needed?

Self-supervised learning is a learning framework that does not use human-labeled datasets to learn a visual representation of the data also known as representation learning. We have been familiar with the task such as classification, detection, and segmentation where a model is trained in a supervised manner which is later used for unseen data. These tasks are normally trained for specific scenarios, for e.g, the ImageNet dataset contains 1000 categories and can only recognize those categories. For categories that are not included in the ImageNet dataset, new annotations need to be done which is an expensive task. Self-supervised makes learning easy as it requires only unlabeled data to formulate the learning task. For training models in a self-supervised manner with unlabeled data, one needs to frame a supervised learning task (also known as _\*pre-text task_). These pre-text tasks can later be used for _\*\*downstream_ tasks such as image classification, object detection, and many more.

_\*Pre-text task: These are the tasks that are used for pre-training  
\*\*Downstream task: These are the task that utilizes pre-trained model or components that can be used to perform tasks such as image recognition, segmentation._

<!-- ![](/images/self-supervised-learning.png) -->
<div align="center"><img src="/images/self-supervised-learning.png"></div>


<p align="center">A general pipeline of self-supervised learning (<a href="https://arxiv.org/pdf/1902.06162.pdf">source</a>) </p>  


## Self-supervised Techniques for Images

Many ideas have been proposed for self-supervised learning on images. A more common methodology or workflow is to train a model in one or multiple pretext tasks with the use of unlabeled data and use that model to perform downstream tasks. Some of the proposed ideas of self-supervised techniques for images are summarized below:

#### Rotation

To learn representation by predicting image rotations, [Gidaris et al.](https://arxiv.org/abs/1803.07728) proposed an architecture where features are learned by training Convolution Nets to recognize rotations that are applied to the image before feeding to the network. This set of geometric transformations defines the classification pretext task that the model has to learn which can later be used for downstream tasks. Geometric transformation is made such that the image is rotated through 4 different angles (0, 90, 270, and 360). This way, our model has to predict one of the 4 transformations that are done on the image. To predict the task, our model has to understand the concept of objects such as their location, their type, and their pose.

<div align="center">
<img src="/images/self-supervised-rotation.png">
 </div>

<p align="center">Illustration of the self-supervised task proposed for semantic feature learning.  
Given four possible geometric transformations, the 0, 90, 180, and 270 degrees rotations,  
a ConvNet model was trained to recognize the rotation that is applied to the image that it gets as input. (<a href="https://arxiv.org/pdf/1803.07728.pdf">source</a>)</p>

More details at: [Unsupervised Representation Learning By Predicting Image Rotations](https://arxiv.org/pdf/1803.07728.pdf)

#### Exemplar

In Exemplar-CNN ( [Dosovitskiy et al., 2015](https://arxiv.org/abs/1406.6909) ), a network is trained to discriminate between a set of surrogate classes. Each surrogate class is formed by applying random data augmentations such as translation, scaling, rotation, contrast and color shifts. While creating surrogate training data:

- N patches of size 32 x 32 pixels are randomly sampled from different images at varying positions. Since, we are interested in patches objects or parts of objects, random patches are sampled only from region containing considerable gradients.
- Each patch is applied with a variety of image transformations. All the resulting transformed patches are considered to be in same surrogate classes.

The pretext task is to discriminate between the set of surrogate class.

<div align="center">
<img src="/images/exemplarCNN-transformation.png">
 </div>
            
<p align="center">Several random transformations applied to one of the  
patches extracted from the STL unlabeled dataset. The original  
patch is in the top left corner (
  <a href="https://arxiv.org/pdf/1406.6909">source</a>)</p>

More details at: [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/pdf/1406.6909.pdf)

#### Jigsaw Puzzle

Another approach of learning visual representation from unlabeled dataset is by training a ConvNet model to solve Jigsaw puzzle as a pretext task which can be later used for downstream tasks. In Jigsaw puzzle task, model is trained to place 9 shuffled patches back to the original position. To place shuffled patches to original position, [Noroozi et al.](https://arxiv.org/pdf/1603.09246) proposed a Context Free Network (CFN) which is a siamese CNN that uses shared weights. The patches are combined in a fully connected layer.

<div align="center">
<img src="/images/jigsaw-puzzle.png">
 </div>

<p align="center">Learning image representations by solving Jigsaw puzzles.  
<ul align="center">(a) The image from which the tiles (marked with green lines) are extracted.  <br>
(b) A puzzle obtained by shuffling the tiles.  <br>
(c) Determining the relative position between the central tile and the top two tiles from the left can be very challenging  
  <a href="https://arxiv.org/pdf/1603.09246.pdf" align="center">source</a></ul></p>

From the set of defined puzzle permutations, one permutation is randomly picked to arrange those 9 patches as per that permutation. This results CFN to return a vector with a probability value for each index. Given those 9 tiles, there will be 9! = 362,880 possible permutations. This creates difficulty in jigsaw puzzles. To control this, the paper proposed to shuffle patches according to a predefined set of permutations and configured the model to predict a probability vector over all the indices in the set.

<div align="center"><img src="/images/cfn_architecture.png"></div>

<p align="center">Context Free Network Architecture (<a href="https://arxiv.org/pdf/1603.09246.pdf">source)</a>  </p>


More details at: [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1603.09246.pdf)

#### Relative Patch Location

This approach by [Doersch et al](https://arxiv.org/pdf/1505.05192), predicts position of second patch of the image that is relative to the first patch. For this pretext task, a network is fed with two input patches and is passed through several convolutional layers. The network produces an output with probability to each of eight image patches. This can be taken as a classification problem with 8 classes where the input patch is assigned to one of these 8 classes to be considered as relative patch to the input patch.

<div align="center"><img src="/images/context_prediction.png"></div>

<p align="center">The algorithm receives two patches in one of these eight  
possible spatial arrangements, without any context, and must then  
  classify which configuration was sampled (<a href="https://arxiv.org/pdf/1505.05192.pdf">source</a>)</p>


More details at: [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192.pdf)

References:

- [The Illustrated Self-Supervised Learning](https://amitness.com/2020/02/illustrated-self-supervised-learning/)
- [Self-supervised Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
- [Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey](https://arxiv.org/abs/1902.06162)
