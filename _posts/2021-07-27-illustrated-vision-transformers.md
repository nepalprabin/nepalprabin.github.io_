# Illustrated Vision Transformers

Ever since Transformer was introduced in 2017, there has been a huge success in the field of Natural Language Processing (NLP). Almost all NLP tasks use Transformers and it’s been a huge success. The main reason for the effectiveness of the Transformer was its ability to handle long-term dependencies compared to RNNs and LSTMs. After its success in NLP, there have been various approaches to its usage for Computer Vision tasks. This paper by Dosovitskiy et al. proposes using the transformer and has achieved some great results in various Computer Vision tasks.

Vison Transformer (ViT) makes use of an extremely large dataset while training the model. While training on datasets such as ImageNet (paper labels ImageNet as a mid-sized dataset), the accuracies of the model fall below ResNets. This is because the Transformer lack inductive bias such as translation equivariance and locality, thus it does not generalize well when trained on insufficient data.

### Overview of Vision Transformer


- Split image into patches
- Provide sequence of linear embeddings of these patches as an input to transformer (flattening the image)
  Here, image patches are treated as the same way as tokens (as in NLP tasks)
- Train the model on image classification in supervised fashion (pre-training with labels)
- Fine-tuning on a downstream dataset

## Method
![](/images/vision_transformer.gif)
<div align="center"> Source: <a href='https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html'>Google AI Blog</a> </div>



Figure above depicts the overview of Vision Transformer. As shown in the figure, given the image, the image is split into patches. These image patches are flattened and passed to transformer encoder a sequence of tokens. Along with patch embeddings, position embedding is also passed as an input to the transformer encoder. Here position embedding is added along with patch embedding to retain positional information. 


## How is an image changed into sequence of vectors to feed into the transformer?
As we know, the [transformer](https://arxiv.org/abs/1706.03762) takes 1D sequence of embeddings as inputs. To match such format, we need to reshape our 2D images. Given the image of size $(H * W * C)$, the paper reshaped into a sequence of flattened 2D patches $x_p \, \varepsilon \, \mathbb{R}^{N*(P^2.C))}$.

An extra learnable embedding is attacheed to the sequence of embedded patches. It is a class embedding (similar to [BERT](https://arxiv.org/abs/1810.04805)'s ```[class]``` token. This extra learnable classification token is used to predict the class of input image which is implemented by a MLP head.

## Components of Vision Transformer
Since Vision Transformer is based on standard transformer architecture, only difference it being used for image tasks rather than for text, components used here is almost the same. Here, we discuss the components used in Vision transformer along with its significance. 

> Side note: If you want to dive deep into transformer, then [here](https://jalammar.github.io/illustrated-transformer/) by Jay Alamamr is a good place to start with.

### Patch embeddings
As the name of the paper "An Image is worth $16*16$ words transformers", the main take away of the paper is the breakdown of images into patches. Given the image $x \varepsilon \mathbb{R}^(H*W*C), it is reshaped into 2D flattened patches: $x_p \, \varepsilon \, \mathbb{R}^{N*(P^2.C))}$,
                        where,
                            N=/frac{H.W}{P^2}, $(P, P)$ is the resolution of each image patch.
                       
