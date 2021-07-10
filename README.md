# Spatial and Temporal Networks for Facial Expression Recognition in the Wild Videos

by Shuyi Mao, Xinqi Fan, Xiaojiang Peng

## Introduction
This repository is for our work Spatial and Temporal Networks for Facial Expression Recognition in the Wild Videos. 

The paper describes our proposed methodology for the seven basic expression classification track of Affective Behavior Analysis in-the-wild (ABAW) Competition 2021. In this task, facial expression recognition (FER) methods aim to classify the correct expression category from a diverse background, but there are several challenges. First, to adapt the model to in-the-wild scenarios, we use the knowledge from pre-trained large-scale face recognition data. Second, we propose an ensemble model with a convolution neural network (CNN), a CNN-recurrent neural network (CNN-RNN), and a CNN-Transformer (CNN-Transformer), to incorporate both spatial and temporal information. Our ensemble model achieved F1 as 0.4133, accuracy as 0.6216 and final metric as 0.4821 on the validation set. 

![](https://github.com/xinqi-fan/ABAW2021/blob/main/figure/pipeline.png)
Figure. Pipeline

## Usage
### Requirement
Python 3.6

PyTorch 1.6

Pandas 1.2


### Download
Clone the repository:
```
git clone https://github.com/xinqi-fan/ABAW2021.git
cd ABAW2021
```

### Prepare data

* Please request from Aff-Wild2 owners.
* Generate labels in csv in data/custom_dataset.py
* Download VGGFace2 weights.


### Train/Validate/Test the model

* Set the weights from trained model and run the following code.

```
# one-to-one
python main_Aff_Wild2_EXPR.py --hpc --model Resnet50Vgg --data_mode static --batch_size 96 --num_workers 8 --learning_rate 5e-4 --save_model --data_folder PATH_TO_DATA --cnn_ckpt weights/xxx.pth
# many-to-many
python main_Aff_Wild2_EXPR_many2many.py --hpc --model CnnRnn --data_mode sequence_by_sequence --batch_size 96 --num_workers 8 --learning_rate 1e-3 --embed_dim 768 --save_model --data_folder PATH_TO_DATA --cnn_ckpt weights/xxx.pth
```


## Citation
To be updated

## Comment
We welcome any pull request for improving the code.

## Contact
To be updated
