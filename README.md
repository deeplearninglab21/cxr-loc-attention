# Localization-Aware Deep Medical Image Classification via Segmentation Driven Gradient-based Attention
The repository contains a method that calculates bounding boxes in real-time during segmentation and integrates Grad-CAM to enhance the loss function for deep classification models.

## Data preparation
Download and extract the NIH chest x-ray data from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345. All compressed packages of images can be downloaded in batch through the .py script contained in the "images" folder. Structure after downloading and extracting all files of images for NIH data:
```
/data/NIH/
  images_01/
    00000001_000.png
    00000001_001.png
  images_02/
    00001336_000.png
    00001337_000.png
  ...
```
Data structure required for model training and evaluation (after running data_prepare.py in data folder):
```
/data/class1_class2/
  train/
    class1/
      img1.png
    class2/
      img2.png
  val/
    class1/
      img3.png
    class/2
      img4.png
  test/
    class1/
      img5.png
    class2/
      img6.png
```
The trained U-Net model is located in 'model' folder
```
/data/Ottawa_masks_512/
  img1.png
  img2.png
  img3.png
```

## Training and evaluation
To run the classification task with cross entropy loss:
```
python model_loss_ce.py 
--path: Path of data
--nclass: Number of classes, 2 or 3 
--gpu: Specify which gpu to use, not required, default is 0
```
To run the classification task with the proposed loss:
```
python model_loss_attent.py 
--dataset: Specify the dateset
--path: Path of data, ../data/class-name/  
--backbone: Backbone model, e.g. pvt, vgg or resent
--task: Specify a classification task: ne, np, nep or neps
--gpu: Specify a gpu if there are more than one
--batch: Specify the batch size
--lamda: Specify a lambda value betweeen 0 to 1
--thresh: Specify a threshold value betweeen 0 to 1
--isAdaptive: A boolean flag indicates whether the value of lambda is adaptive, e.g. 0.9
```
This should give a .txt file (The file name will be generated based on the values of lambda and threshold) with performances on validation and testing sets respectively.

__Examples__

To train a PVT model with cross entropy loss, 3 classes:
```
python pvt_loss_ce.py --path ../data/nofind_effusion_pneumothorax/ --nclass 3
```
To train a VGG16 model with proposed loss, dataset is Ottawa, multi-class (No Finding, Effusion, Pneumothorax, and Subcutaneous emphysema), gpu is 0, lambda is adaptive, threshold is 0.9:
```
python model_loss_attent.py \
--dataset ottawa \
--path ../data/Ottawa/nf_e_p_s/ \
--backbone vgg \
--task neps \
--gpu 0 \
--isAdaptive \
--thresh 0.9

```
To train a ResNet50 model with proposed loss, dataset is NIH, binary classes (No Finding vs Effusion), gpu is 1, lambda is 0.25, threshold is 0.7:
```
python model_loss_attent.py \
--dataset nih \
--path ../data/NIH/nofind_effusion/ \
--backbone resnet \
--task ne \
--gpu 1 \
--lamda 0.25 \
--thresh 0.7
```

### Note

Ottawa dataset is undisclosed;
To make sure the U-Net model has been trained and in folder 'model/unet_model.pt'.
