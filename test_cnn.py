# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:14:24 2020

@author: singh
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow.keras.models import load_model
from tqdm import tqdm
from impath_generator import get_impaths
import cv2, random, argparse, numpy as np, tensorflow as tf

# Prevent total GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help="path to the input model")
ap.add_argument('-i', '--images', required=True, help="path to the input images directory")
ap.add_argument('-p', '--path', required=True, help="path to the output tested images")
args = vars(ap.parse_args())

print("[INFO] Loading model...")
model = load_model(args['model'])

print("[INFO] Loading prediction images...")
impaths = get_impaths(args['images'])
random.shuffle(impaths)

print("[INFO] Reading class CSV...")
classes = open('classes.csv').read().strip().split('\n')[1:]
classes = [l.split(',')[1] for l in classes]
    


# For Upscaling/Downscaling images
def im_scale(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height =  int(image.shape[0] * scale_factor)
    dims = (width , height)
    image = cv2.resize(image, dims, interpolation=cv2.INTER_CUBIC)
    
    return image


print("[INFO] Predicting...")
for i, impath in tqdm(enumerate(impaths)):
    image = cv2.imread(impath)
    image = cv2.resize(image, (160, 160))
    
    # Scaling image to [0,1]
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Making predictions using our trained CNN
    predY = model.predict(image)
    idx = predY.argmax(axis=1)[0]
    idx_proba = np.amax(predY)
    label = classes[idx]
    
    image = cv2.imread(impath)
    image = im_scale(image, 1.3)
    
    textLabel = f'{label} : {idx_proba:.4f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 2
    textSize, baseLine = cv2.getTextSize(textLabel, fontFace=font, fontScale=scale, thickness=thickness)
    cv2.rectangle(image, (14, 40), (16 + textSize[0], 55 + textSize[1]), (255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(image, textLabel, (15, 45 + baseLine*2), font, scale, (0, 0, 0), thickness)
    
    if os.path.exists(args['path']):
        p = os.path.sep.join([args['path'], f'{i+1}.png'])
    else:
        os.mkdir(args['path'])
        p = os.path.sep.join([args['path'], f'{i+1}.png'])
        
        
    cv2.imwrite(p, image)
    
    
print("[INFO] Result images saved in {} directory...".format(args['path']))    
    
    
    
    
