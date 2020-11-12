# multidimensional_vector-estimation

Code for training/estimate latent variable of [Font Design GAN][]

## Overview

This estimator estimates the latent variable of Font Design GAN.  

The reason why construct the estimator is that we want to make the dataset for training Book Cover Title Generation Network.

## Prepare

To train this estimator, train Font Design GAN and get the pair of the latent variable and the alphabet generated from the variable.
Probably you can view [Font Design GAN fork][] repository forked from [Font Design GAN][]


## Experiment

We tried to estimate the latent variable from 100000 alphabet images genetated by Font Design GAN.

The track of training and validation loss is here.

As the result shows, the network can't learn the relationship between 
font style and latent variable. 

## Conclusion

[Font Design GAN]: https://github.com/uchidalab/fontdesign_gan "Font Design GAN"
[Font Design GAN fork]: https://github.com/Taylister/fontdesign_gan "Font Design GAN fork"
