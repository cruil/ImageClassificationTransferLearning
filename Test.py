# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:21:52 2019

@author: DMarudhu
"""

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.inception_v3 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input

# other imports
import numpy as np
import glob
import json
import shutil
import pickle as cPickle
import os

# load the user configs
with open('conf.json') as f:    
	conf = json.load(f)

# config variables
model_name = conf["model"]
weights = conf["weights"]
include_top = conf["include_top"]
train_path = conf["train_path"]
features_path = conf["features_path"]
labels_path = conf["labels_path"]
test_size = conf["test_size"]
model_path = conf["model_path"]
seed = conf["seed"]
features_path = conf["features_path"]
results = conf["results"]
classifier_path = conf["classifier_path"]
num_classes = conf["num_classes"]
predictions_path=conf["pred_path"]



loaded_model = cPickle.load(open(classifier_path, 'rb'))
base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, output=base_model.layers[-1].output)
image_size = (299, 299)



# path to training dataset
train_labels =  os.listdir(train_path)


features=[]
cur_path = 'Test'
for test_path in glob.glob(cur_path + "/*.jpg"):
	#load = i + ".png"
	#print ("[INFO] loading", test_path,"image ")
	img = image.load_img(test_path, target_size=image_size)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	feature = model.predict(x)
	flat = feature.flatten()
	features.append(flat)
	preds = loaded_model.predict(features)
	print (preds)
#	show_image = cv2.imread(test_path)
	#show_image = cv2.resize(show_image, (500, 500)) 
for i in range (0,len(train_labels)):
	if preds[len(preds)-1]== i:
		shutil.copy2(test_path, predictions_path+'/'+train_labels[i])
#            	elif preds[len(preds)-1] == 1: 
#            		shutil.copy2(test_path,'Predictions\\Cats')
#            	elif preds[len(preds)-1] == 2: 
#            		shutil.copy2(test_path,'Predictions\\Dogs')
#            	elif preds[len(preds)-1] == 3: 
#            		shutil.copy2(test_path,'Predictions\\Not_Birds')
#            	del preds

	#y_classes = preds.argmax(axis=-1)
#disease = preds
	#print (show_image)
#	print(preds) m
	#cv2.putText(show_image, preds,cv2.FONT_HERSHEY_SIMPLEX)
	#cv2.putText(show_image, preds, (40,50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
	#cv2.imshow("result",show_image)
	#cv2.waitKey(0)
print("Results are classified and saved")



