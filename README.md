# CRNN Pytorch

This is a Pytorch implementation of a Deep Neural Network for scene text recognition. It is based on the paper ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2016), Baoguang Shi et al."](http://arxiv.org/abs/1507.05717) forked from [this repo](https://github.com/GitYCC/crnn-pytorch)


## Quick Demo

```command
$ pip install -r requirements.txt
$ python predict.py --i novel.jpg --model weights/crnn.pt
```
Result:

```bash
device: cuda
Predicted Text: novel
```

![novel](./novel.jpg)



## Train

Download Synth90k dataset

```command
$ cd data
$ bash download_synth90k.sh
```

```
@InProceedings{Jaderberg14c,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition",
  booktitle    = "Workshop on Deep Learning, NIPS",
  year         = "2014",
}

@Article{Jaderberg16,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Reading Text in the Wild with Convolutional Neural Networks",
  journal      = "International Journal of Computer Vision",
  number       = "1",
  volume       = "116",
  pages        = "1--20",
  month        = "jan",
  year         = "2016",
}
```

## Pretrained Model

I've per-trained a CRNN model on Synth90k dataset although it's not fully trained. So train it yourself if you want to get acceptable results.

### Evaluate the model on the Synth90k dataset

```bash
$ python evaluate.py
```

## Train your model

You could adjust hyper-parameters in `config.py`.

And train crnn model,

```command
$ python train.py
```

## Reference
This repo is a cleaner and more compact implementation of [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch) by [GitYCC](https://github.com/GitYCC)
