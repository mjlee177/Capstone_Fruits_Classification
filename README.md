# Capstone_Fruits_Classification

## Author

Matthew J. Lee<br>
BS/MS Mech/Aero Eng UC Davis, MBA Tepper, Tech Proj Mgmt UC Berkeley, ML & AI UC Berkeley<br>
Aerospace Engineer, Technical Project Manager, AI Computer Vision Engineer, Entrepreneur<br>
https://mattjlee.info

## Introduction

The goal of this capstone is to classify 10,000 pictures of 5 different fruits. We explore multiple models (Conv2D/Maxpooling2D, MobileNetV2, EfficientNetV2B0, EfficientNetB7, ResNet152V2, InceptionV3, Xception, ConvNeXtBase) and then hypertune to report the best accuracy.

## Dataset

Kaggle dataset link: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification/

The fruits classification dataset is from Kaggle.  It contains 10,000 images with an even number (2,000 each) of 5 different types of fruits:
- apples
- bananas
- grapes
- mangoes
- strawberries

The pictures are of various shapes, sizes, colors, and lighting conditions.  Some are pictures taken with a camera, some are computer generated, and some are drawn by hand.  All types of scenes and angles are taken of the fruits, including whole, sliced, peeled, bitten, plucked, on the tree/ vine, and arranged on dishes.  Some pictures even have false colors (like a blue apple) or are in black-and-white.  What is consistent among these images is that the fruits are true to shape, meaning there's no mashed banana, apple sauce, grape juice, or other byproducts.  If there is a picture of a byproduct (e.g. strawberry cake), the fruit is there as well.  Some images are of the full fruit, some are in bunches, some are only partially on the image or some of the fruit is not exactly true to shape because it's dipped in chocolate or something similar.<br>

![Hand drawn apple](images/Apple%20(1).png)
Hand drawn apple<br>
![Computer generated strawberry](images/Strawberry%20(883).jpeg)
Computer generated strawberry<br>
![Sliced and arranged apples](images/Apple%20(21).jpeg)
Sliced and arranged apples<br>
![Blue apple](images/Apple%20(80).jpeg)
Blue apple<br>
![Strawberry cake](images/Strawberry%20(132).jpeg)
Strawberry cake<br>
![Chocolate dipped strawberries](images/Strawberry%20(775).jpeg)
Chocolate dipped strawberries<br>
![Banana mostly out of the picture](images/Banana%20(3071).jpeg)
Banana mostly out of the picture<br>

The pictures have varying dimensions but all are 96 dpi vertical and horizontal resolution, have 24 bit depth, and are in jpeg format.

The data is split into 97% training, 2% validation, and 1% testing.  Since there is a total of 10,000 pictures, this means that for each of the 5 fruit classes, there are 1940 training, 40 validation, and 20 testing images.  This ensures that distribution of classes is consistent across all three sets and that the model is trained on a representative sample.

### Modeling

 We will use a baseline model made up of Conv2D/Maxpooling2D layers and 7 pre-trained models.  Models summaries:

 Model              | Gen / Year       | Input Size | Params (M) | ImageNet Top-1 Acc. | Speed / Size      | Best For                                 |
|--------------------|------------------|------------|------------|----------------------|-------------------|-------------------------------------------|
| MobileNetV2        | Classic / 2018   | 224×224    | ~3.4M      | ~71.8%               | Very fast, lightweight | Mobile, embedded, small datasets      |
| EfficientNetV2B0   | Next-Gen / 2021  | 224×224    | ~7.1M      | ~82.3%               | Fast, efficient    | Fast training, limited resources          |
| ResNet152V2        | Classic / 2016   | 224×224    | ~60.2M     | ~78.3%               | Slower, large      | Very deep, general-purpose                |
| EfficientNetB7     | Classic / 2019   | 600×600    | ~66M       | ~84.4%               | Heavy compute      | Max accuracy, strong GPU                  |
| InceptionV3        | Classic / 2015   | 299×299    | ~23.8M     | ~78.8%               | Medium             | Multi-scale features, strong all-arounder |
| Xception           | Classic / 2017   | 299×299    | ~22.9M     | ~79.0%               | Medium             | Depthwise separable convs, efficient      |
| ConvNeXtBase       | Next-Gen / 2022  | 224×224    | ~88M       | ~83.1%               | High memory usage  | Modern CNN rivaling ViT                   |

Here is a quick comparison chart with each of these models:


|

## Summary of Conclusions
