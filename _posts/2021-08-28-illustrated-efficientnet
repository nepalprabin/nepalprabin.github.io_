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
Normally ConvNets opt for finding best network architecture, model scaling expand the network length, width and resolution without changing the model architecture that is predefined in the baseline network. 

### Scaling Dimensions
Finding the optimal depth ```d```, width ```w``` and resoution ```r``` is a difficult problem due to which ConvNets mostly scale in one of these dimensions.

### Depth
Scaling Convolution Neural Networks by depth is the most conventional method. Starting from AlexNet (2012), many neural network architectures went deep down the network to increase the model performance. This works to some extent as going deeper into the network helps the model to capture complex features. But, if we go deeper, we may face a vanishing gradient problem. Vanishing gradient problem had been solved by introducing skip connections as in ([Resnet](https://arxiv.org/abs/1512.03385)), but the training accuracy does not seem to change.

From the paper:
> However, deeper networks
are also more difficult to train due to the vanishing gradient
problem (```Zagoruyko & Komodakis, 2016```). Although several techniques, such as skip connections (```He et al., 2016```)
and batch normalization (```Ioffe & Szegedy, 2015```), alleviate
the training problem, the accuracy gain of very deep network
diminishes: for example, ResNet-1000 has similar accuracy
as ResNet-101 even though it has much more layers.

### Width
Small models such as [MobileNets](https://arxiv.org/abs/1704.04861) scale network width to train deep network models. With more width in the network, the network can capture fine-grained features and they are easier to train. But once we keep on expanding the width of the network, it is difficult to capture higher level features and the accuracy saturates at some point as we keep on increasing the width of the network.

### Resolution
With increasing the resolution of the images, convolution networks capture more features (patterns). Though the accuracy of the model increases with increase in the resolution, the accuracy drops down for very high resolutions.
