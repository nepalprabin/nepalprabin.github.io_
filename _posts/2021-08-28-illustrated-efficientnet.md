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
