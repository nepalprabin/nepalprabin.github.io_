# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
Convolution Neural Networks(CNN) has been a go-to architecture for Computer Vision tasks. With the availability of much compute resources, huge models have been introduced
frequently in terms of number of layers. In 2012, [deep learning network](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) proposed by Alex Krizhevsky et. al., 
took machine learning community by storm and after this many architectures have been proposed to obtain the state of the art results in many machine learning tasks. 
Some approaches increased number of layers to increase the accuracy of model while other increased the resolution of images to achieve the same. **EfficientNet** proposed model scaling method to uniformly scale 
depth, width and resolution for better performance of the model.

From the paper abstract:
>In this paper, we systematically study model scaling and identify that
carefully balancing network depth, width, and resolution can lead to better performance. Based
on this observation, we propose a new scaling
method that uniformly scales all dimensions of
depth/width/resolution using a simple yet highly
effective compound coefficient.

EfficientNet basically used Neural Architecture Search(NAS) to design a new baseline network and scale it up to get its family models known as **EfficientNets**.
This approach achieved more accuracy and efficiency than other ConvNets. In fact, ```EfficientNet-B7``` achieves state-of-the-art ```84.3%``` top-1 accuracy on ImageNet.

## Compound Model Scaling
Normally ConvNets opt for finding best netwoek architecture, model scaling expand the network length, widht and resolution without changing the model architecture that is predefined in the baseline network. 

### Scaling Dimensions
Finding the optimal depth ```d```, width ```w``` and resoution ```r``` is a difficult problem due to which ConvNets mostly scale in one of these dimensions.

### Depth
Scaling Convolution Neural Networks by depth is the most conventional methods. Starting from the AlexNet (2012), many neural network architecture went deep down the network to increase the model performance. This works to some extent as going deeper into the network helped model to capture complex features. But, if we go more deep, model becomes harder to train as we may face vanishing gradient problem. Vanishing gradient problem had been solved by introducing skip connections as in ([Resnet](https://arxiv.org/abs/1512.03385)), but the training acccuracy doesnot seem to change.

From the paper:
> However, deeper networks
are also more difficult to train due to the vanishing gradient
problem (```Zagoruyko & Komodakis, 2016```). Although several techniques, such as skip connections (```He et al., 2016```)
and batch normalization (```Ioffe & Szegedy, 2015```), alleviate
the training problem, the accuracy gain of very deep network
diminishes: for example, ResNet-1000 has similar accuracy
as ResNet-101 even though it has much more layers.

### Width

