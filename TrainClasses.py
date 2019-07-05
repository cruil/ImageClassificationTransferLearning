# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:20:20 2019

@author: DMarudhu
"""

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import h5py
import os
import json
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle as cPickle


# load the user configs
with open('conf.json') as f:    
    conf = json.load(f)

# config variables
weights = conf["weights"]
include_top = conf["include_top"]
train_path = conf["train_path"]
features_path = conf["features_path"]
labels_path = conf["labels_path"]
test_size = conf["test_size"]
model_path = conf["model_path"]
seed = conf["seed"]
features_path = conf["features_path"]
classifier_path = conf["classifier_path"]

#base model
base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, output=base_model.layers[-1].output)
image_size = (299, 299)

# path to training dataset
train_labels =  os.listdir(train_path)

# encode the labels
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
try:
    for i, label in enumerate(train_labels):
        cur_path = train_path + "/" + label
        count = 1
        for image_path in glob.glob(cur_path + "/*.jpg"):
            img = image.load_img(image_path, target_size=image_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            flat = feature.flatten()
            features.append(flat)
            labels.append(label)
            #print ("[INFO] processed - " + str(count))
            count += 1
        print ("[INFO] completed label - " + label)
except OSError:
    pass


    
# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# save model and weights
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
    json_file.write(model_json)

# save weights
model.save_weights(model_path + str(test_size) + ".h5")
print("[STATUS] saved model and weights to disk..")

print ("[STATUS] features and labels saved..")


# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string = h5f_label['dataset_1']

features = np.array(features_string)
labels = np.array(labels_string)

h5f_data.close()
h5f_label.close()
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                np.array(labels),
                                                                test_size=test_size,
                                                                random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print ("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)

# dump classifier to file
print ("[INFO] saving model...")
f = open(classifier_path, "wb")
f.write(cPickle.dumps(model))
f.close()
