import csv
import os
import random
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, AveragePooling2D

# %matplotlib inline

# command line flags
try:
    flags
except Exception:
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('data_dir', 'dataset', 'Directory of the dataset and driving log.')
    flags.DEFINE_string('camera_correction', 0.15, 'Steering correction for the side camera images.')
    flags.DEFINE_string('batch_size', 64, 'Batch_size of training.')
    flags.DEFINE_string('epoch', 3, 'Epoch of training.')
    flags.DEFINE_string('model', 'model.h5', 'Saved model file.')


with open(os.path.join(FLAGS.data_dir, 'driving_log.csv'), 'r') as f:
    reader = csv.reader(f)
    samples = []
    for row in reader:
        # Drop half of 0 steering samples
        if float(row[6]) != 0 and random.random() < 0.5:
            samples.append(row)

# Split data set to train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=FLAGS.batch_size):
    def process_angle(angle, speed):
        """
        All the angles are between -1.2 to 1.2, cut them to [-1, 1] one hot data.
        And I have also increase the angle when I have speed down to pass the curve.
        """
            
        if angle > 1.0:
            return 1.0
        elif angle < -1.0:
            return -1.0
        else:
            return angle

    def fix_path(path):
        """ Because I've recoed the data in Windows, so need fix the image path in the log file. """
        path = path.strip()
        path = path[path.find('IMG'):]
        path = os.path.join(*path.split('\\'))
        return path

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            actions = []
            for batch_sample in batch_samples:
                speed = float(row[6])
                steering_center_angle = float(row[3])

                img_center = imread(os.path.join(FLAGS.data_dir, fix_path(row[0])))
                img_left = imread(os.path.join(FLAGS.data_dir, fix_path(row[1])))
                img_right = imread(os.path.join(FLAGS.data_dir, fix_path(row[2])))
                
                # Add images, fliped images and actions to data set
                images.extend([img_center, img_left, img_right])
                images.extend([np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
                actions.extend([
                    process_angle(steering_center_angle, speed),
                    process_angle(steering_center_angle + FLAGS.camera_correction, speed),
                    process_angle(steering_center_angle - FLAGS.camera_correction, speed)
                ])
                actions.extend([
                    process_angle(-steering_center_angle, speed),
                    process_angle(-steering_center_angle - FLAGS.camera_correction, speed),
                    process_angle(-steering_center_angle + FLAGS.camera_correction, speed)
                ])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(actions).reshape(-1, 1)
            yield shuffle(X_train, y_train)

                                        
def CNN():
    """ My CNN based on VGGNet. """
    model = Sequential()
    model.add(Cropping2D(cropping=((64, 24), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(72, 320, 3)))
    # input: 72x320 images with 3 channels
    # Using one average pooling to change the image size smaller
    model.add(AveragePooling2D(pool_size=(2, 4)))

    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(36, 5, 5, border_mode='valid', input_shape=(72, 320, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('tanh'))
    
    model.compile(loss='mse', optimizer='adam')
    return model


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

if os.path.exists(FLAGS.model):
    model = load_model(FLAGS.model)
else:
    model = CNN()
    
# Because I've flip all the image and 3 directions, so the size of data set must be multiple by 6
history_object = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples) * 6,
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples) * 6,
    nb_epoch=FLAGS.epoch,
    verbose=1
)

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

# Save the model
model.save(FLAGS.model)
