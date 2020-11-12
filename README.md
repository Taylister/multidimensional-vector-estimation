# multidimensional-vector-estimation

Code for training/estimate latent variable of [Font Design GAN][]

## Overview

This estimator estimates the latent variable of Font Design GAN.  

The reason why construct the estimator is that we want to make the dataset for training Book Cover Title Generation Network.

## Prepare

To train this estimator, train Font Design GAN and get the pair of the latent variable and the alphabet generated from the variable.

Probably you can view [Font Design GAN fork][] repository forked from [Font Design GAN][] as the reference.

## Train

You have to prepare a dataset in the following format.

```
  data/ 
      |____images/ 
      |       |____XXXX.jpg # .png format is also acceptable.
      |       |____0000.jpg
      |       |____....
      |____vectors/ 
              |____XXXX.npy
              |____0000.npy  
              |____....
```

The image file name sould be the same as the latent variable file name correspond to image file. 

After deploy the data above format, just `bash make_dataset.sh`.

The bash file splits the data between train data and test data.

## Experiment

We tried to estimate the latent variable from 100000 alphabet images genetated by Font Design GAN.

The track of training and validation loss is here.

[Not ready for now]

As the result shows, the network can't learn the relationship between 
font style and latent variable. 

## Conclusion

The task that estimate the latent variable of Font Design GAN (100dim) is too hard to utilize the purpose.  


[Font Design GAN]: https://github.com/uchidalab/fontdesign_gan "Font Design GAN"
[Font Design GAN fork]: https://github.com/Taylister/fontdesign_gan "Font Design GAN fork"
