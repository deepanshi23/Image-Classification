import json, os, re, sys, time
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

import time
model_path = "C://Users//Deepanshi//Documents//iimc//resnet50_final_v2.h5"
#model_path = "C://Users//Deepanshi//Documents//iimc//inceptionv3_best.h5"
print('Loading model:', model_path)
t0 = time.time()
model = load_model(model_path)
t1 = time.time()
print('Loaded in:', t1-t0)

##for creating the contingency table - 
root = 'C://Users//Deepanshi//Documents//iimc//data//Test//Class1//'
Test_v1_Class1 = []
for root, _, filenames in os.walk(root):
    for filename in filenames:
        Test_v1_Class1.append(os.path.join(root, filename))

preds_v1_Class1 = []
for filename in Test_v1_Class1:
    preds_v1_Class1.append(predict(filename,model))
count = 0

for i in range(len(preds_v1_Class1)):
    if((preds_v1_Class1[i][0,0])>(preds_v1_Class1[i][0,1])):
        count = count+1
        
print("Correct Predictions for Class 1 are",count," and Total Images in Class 1 are",len(preds_v1_Class1))

root = 'C://Users//Deepanshi//Documents//iimc//data//Test//Class2//'
Test_v1_Class2 = []
for root, _, filenames in os.walk(root):
    for filename in filenames:
        Test_v1_Class2.append(os.path.join(root, filename))

preds_v1_Class2 = []
for filename in Test_v1_Class2:
    preds_v1_Class2.append(predict(filename,model))

count = 0
for i in range(len(preds_v1_Class2)):
    if((preds_v1_Class2[i][0,0])<(preds_v1_Class2[i][0,1])):
        count = count + 1
        
print("Correct Predictions for Class 2 are",count," and Total Images in Class 2 are",len(preds_v1_Class2))


lst = [item2[0] for item2 in [item[0] for item in preds_v1_Class1]]
lst.extend([item2[0] for item2 in [item[0] for item in preds_v1_Class2]])
a = [(1 if item > 0.5 else 0) for item in lst]
y_test = [1]*180
y_test.extend([0]*190)
print(y_test)

##roc auc score
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,lst)

##log loss
from sklearn.metrics import log_loss
log_loss(y_test,a)
