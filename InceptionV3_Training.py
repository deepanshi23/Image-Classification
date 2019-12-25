import math, json, os, sys
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import time

DATA_DIR = 'C://Users//Deepanshi//Documents//iimc//data'
TRAIN_DIR = os.path.join(DATA_DIR, 'Train/')
VALID_DIR = os.path.join(DATA_DIR, 'Valid/')
SIZE = (224, 224)
BATCH_SIZE = 16

num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

gen = keras.preprocessing.image.ImageDataGenerator()
val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)4

batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

model = keras.applications.inception_v3.InceptionV3()

classes = list(iter(batches.class_indices))
model.layers.pop()
for layer in model.layers[:-2]:
    layer.trainable=False
    
last = model.layers[-1].output

x = Dense(len(classes), activation="softmax")(last)

finetuned_model = Model(model.input, x)
finetuned_model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
for c in batches.class_indices:
    classes[batches.class_indices[c]] = c
finetuned_model.classes = classes

early_stopping = EarlyStopping(patience=5)
checkpointer = ModelCheckpoint('inceptionv3_best.h5', verbose=1, save_best_only=True)

#training for learning rate = 0.001, batch size = 16, epoch = 20, training last 2 layers of inceptionv3 + softmax layer
finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=20, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
