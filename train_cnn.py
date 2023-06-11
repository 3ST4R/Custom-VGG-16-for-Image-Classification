# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:15:49 2020

@author: singh
"""

# Setting environment variables
import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Turn off Tensorflow logger
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 for processing
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU for processing

# Set the matplotlib backend so that figures can be saved in background
import matplotlib
matplotlib.use('Agg')

# Import the necessary packages
from dogcatnet import ConvNN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from impath_generator import get_impaths
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import tensorflow as tf, argparse, numpy as np, math, cv2
# import pandas as pd

# Prevent total GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Making a custom Yes/No switch
def switch_0_1(value):
    if value.lower() in ['0', 'false', 'n', 'no', None]:
        return False
    elif value.lower() in ['1', 'true', 'y', 'yes']:
        return True
    else:
        raise ValueError("Invalid boolean value")


# Argument parser to pass via command line
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help="path to input dataset")
ap.add_argument('-a', '--augment', type=switch_0_1, nargs='?', const=True, default=False, help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument('-p', '--plot', type=str, default='plot.png', help="path to output loss/accuracy plot")
ap.add_argument('-b', '--batchsize', type=int, default=32, help="Number of training samples processed in a single batch")
ap.add_argument('-m', '--model', default='./dogcatcnn_model.hdf5', help="path to the output model")
ap.add_argument('-e', '--earlystopping', type=switch_0_1, nargs='?', const=True, default=False, help="whether or not to use early stopping")
ap.add_argument('-c', '--checkpoint', type=switch_0_1, nargs='?', const=True, default=False, help="whether or not to use model checkpointing")
ap.add_argument('-eval', '--evaluate', default='./evaluation.txt', help="path to the model evaluation text file")
ap.add_argument('-dcy', '--lrdecay', type=float, default=0.01, help="Learning decay rate")

args= vars(ap.parse_args())

# Setting the input image dimensions
image_dims = (160, 160)

# Defining some useful functions to make things easier to understand   
def load_imdata(path):
    impaths = get_impaths(path, train_justify=True)
    data, classes = [], []
    
    for impath in tqdm(impaths):
        # Extracting class label from filename
        # Load image and resizing it to image_dims
        label = impath.split(os.path.sep)[-2]
        image = cv2.imread(impath)
        image = cv2.resize(image, image_dims)
        
        # Appending it to img_data and classes list
        data.append(image)
        classes.append(label)
    
    # Convert the lists into numpy arrays    
    data, classes = np.array(data), np.array(classes)
        
    return data, classes

# Set up exponential decay to the learning rate of the optimizer
def exp_decay_1(epoch, initial_lr):
    '''Formula => a = b * e^(-kt)
    a = new learning rate, b = initial learning rate
    k = decay rate, t = number of epochs elapsed
    '''
    
    decay_rate = args['lrdecay']
    new_lr = initial_lr * math.exp(-decay_rate * epoch)
    
    return new_lr

def exp_decay_2(epoch, initial_lr):
    '''Formula => a = b(1 - k)^t
    a = new learning rate, b = initial learning rate
    k = decay rate, t = number of epochs elapsed
    '''
    if initial_lr <= 0.002:
        decay_rate = 0
    else:
        decay_rate = args['lrdecay']
    new_lr = initial_lr * (1 - decay_rate) ** epoch
    
    return new_lr

# Loading images from the dataset,
# listing them into img_data and classes
print('[INFO] Loading images...')
imdata, classes = load_imdata(args['dataset'])


# Getting the class indices
class_labels = []
for x in classes:
    if x not in class_labels:
        class_labels.append(x)

# Creating CSV for class names for future use (only once)
# pd.DataFrame(np.array([range(len(class_labels)), class_labels]).transpose(), 
#                   columns=['ClassId', 'ClassName']).to_csv('classes.csv', index=None)


# Scale image data to the range of [0, 1]
imdata = imdata.astype('float32') / 255.0


# Encode the classes list as integers and 
# then OneHotEncode it
le_class = LabelEncoder()
classes = le_class.fit_transform(classes)
classes = to_categorical(classes, len(class_labels))


print("\n[INFO] Found {} images belonging to {} classes".format(imdata.shape[0], len(class_labels)))


# Split the img_data into train and test set
trainX, testX, trainY, testY = train_test_split(imdata, classes, test_size=0.2, random_state=0)

print("\n[INFO] Splitting the images into {} images for training set & {} images for test set".\
      format(trainX.shape[0], testX.shape[0]))

    
    
# Data Generation/Augmentation
aug_set = ImageDataGenerator()

if args['augment']:
    print("\n[INFO] Performing 'on the fly' data augmentation")
    aug_set = ImageDataGenerator(
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

val_set = ImageDataGenerator()



# Initializing the model
NUM_EPOCHS = 60
INIT_LR = 1e-1
BS = args['batchsize']



# Setting optimizer
opt = Adadelta(lr=INIT_LR)


# Compiling the model
print("\n[INFO] Compiling the model...\n")
# model = ConvNN.build(160, 160, 3, 1)
# model = ConvNN.build_simple(image_dims)
model = ConvNN.build_soa(image_dims, nb_layers=4)


# Summary of the model
model.summary()
print("\n[INFO] Verify the model, press any key to continue...")
os.system("PAUSE >null")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# Set up Learning Rate Scheduler callback for the custom decay function
lrate = LearningRateScheduler(exp_decay_2, verbose=1)


# Stop the training if validation loss starts to increase
if args['earlystopping']:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
else:
    es = None

# Save the model with best test set accuracy
if args['checkpoint']:
    ckpt = ModelCheckpoint('dogcatnet_cnn.hdf5', monitor='val_accuracy', mode='max', 
                                save_best_only=True, verbose=1)
else:
    ckpt = None


callback_list = [es, ckpt, lrate]
callback_list = [x for x in callback_list if x]

print(f"\n[INFO] Training network for {NUM_EPOCHS} epochs\n")

H = model.fit_generator(
                    aug_set.flow(trainX, trainY, batch_size=BS), 
                    steps_per_epoch=trainX.shape[0] // BS,
                    epochs=NUM_EPOCHS,
                    validation_data=val_set.flow(testX, testY, batch_size=BS),
                    validation_steps=testX.shape[0] // BS,
                    callbacks=callback_list,
                    verbose=1)
			
	

# Evaluating the model's performance
print("\n[INFO] Evaluating the network...\n")
TRAIN_LOSS, TRAIN_ACC = model.evaluate(trainX, trainY, verbose=0)
TEST_LOSS, TEST_ACC = model.evaluate(testX, testY, verbose=0)
print("Training Accuracy : {:.2f}% , Test Accuracy : {:.2f}%".format(TRAIN_ACC*100, TEST_ACC*100))
print("Training Loss     : {:.4f} , Test Loss     : {:.4f}".format(TRAIN_LOSS, TEST_LOSS))
with open(args['evaluate'], 'w') as eval_file:
    print("Training Accuracy : {:.2f}% , Test Accuracy : {:.2f}%".format(TRAIN_ACC*100, TEST_ACC*100), file=eval_file)
    print("Training Loss     : {:.4f} , Test Loss     : {:.4f}".format(TRAIN_LOSS, TEST_LOSS), file=eval_file)

# Saving the model
model.save(os.path.abspath(args['model']))



# Loss/Acc Graph Plot
if args['earlystopping']:
    STOPPED_EPOCH = es.stopped_epoch + 1
    if STOPPED_EPOCH == 1:
        STOPPED_EPOCH = NUM_EPOCHS
else:
    STOPPED_EPOCH = NUM_EPOCHS
    	

N = np.arange(0, STOPPED_EPOCH)
plt.style.use('ggplot')
plt.figure(figsize=(10.5, 14), dpi=100) # Figure size in 1e-2

# 1st Sub plot on the graph
plt.subplot(3, 1, 1) # 3 rows, 1 column, index=1
plt.title("Loss/Accuracy Graph for Train/Validation")
plt.plot(N, H.history['loss'], label='Train')
plt.plot(N, H.history['val_loss'], label='Validation')
plt.ylabel("Loss")
plt.legend(loc='upper right')

# 2nd Sub plot on the graph
plt.subplot(3, 1, 2) # 3 rows, 1 column, index=2
plt.plot(N, H.history['accuracy'], label='Train')
plt.plot(N, H.history['val_accuracy'], label='Validation')
plt.ylabel("Accuracy")
plt.legend(loc='lower right')

# 3rd Sub plot on the graph
plt.subplot(3, 1, 3) # 3 rows, 1 column, index=3
plt.plot(N, H.history['lr'], label='Learning Rate')
plt.ylabel("Learning rate")
plt.xlabel("Epoch #")
plt.legend(loc='upper right')
plt.savefig(args['plot'])








