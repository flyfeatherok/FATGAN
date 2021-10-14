# Facial Attributes Translation with Attention Consistency and Self-calibration

we propose a generative adversarial network in this paper, which the attention is constrained by a new loss function named attention consistency loss and guided by self-calibration convolution. Stronger constraints on attention improve the quality of facial attributes translation results and self-calibration convolution effectively improves attention coverage.

Paper URL is coming

## Requirements

- [Windows or Linux]
- [Tensorflow ( 1.14.0)](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

## Dataset Preparation

    ├── main.py
    ├── model
        ├── checkpoint
    ├── CelebA
       ├── celeba_128
            ├── 000001.jpg 
            ├── 000002.jpg
            └── ...
       ├── Anno
            ├── list_attr_celeba.txt (For attribute information) 
    ├── test_img
        ├── a.jpg (The test image that you wanted)
        ├── b.png
        └── ...

## Train

    python main.py --phase train

## Test

	python main.py --phase test

- There will be some trackbar for attribute vactor contrl. The vaule of trackbar [0, 200] means the vaule of attribute vactor [-1, 1].

## Pretrained model

- Download checkpoint for [128x128](https://pan.baidu.com/s/10FHomVjkceuh0kJXIQsb9w)cbx6
