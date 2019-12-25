# Image-Classification
This repository contains the code files to identify the presence of human beings in Earthquake debris images using CNN.
The dataset has been synthetically expanded using augmentation techniques - rotation, translation, rescaling, flipping, shearing, 
stretching and adding random noise.

CNN models - Resnet50 and InceptionV3 have been applied on the dataset.

The models have been compared on various metrics - 
1. Overall accuracy
2. Class wise accuracy
3. Precision
4. Recall
5. AUC Score
6. F1 score

The files should be executed in the following order - 
1. data_augmentation.py
2. Resnet_Training.py / InceptionV3_Training.py
3. Model_Predictions.py
