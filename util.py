import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

BATCH_SIZE = 32

'''
Transforms each image in dataset by pasting it on a 75x75 canvas at random locations.
'''
def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0 , 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0 , 48, dtype=tf.int32)
    image = tf.reshape(image, (28,28,1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32)/255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
   
    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])
  
'''
Loads and maps the training split of the dataset using the map function. Note that we try to load the gcs version since TPU can only work with datasets on Google Cloud Storage.
'''
def get_training_dataset():
      
      
    dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    #dataset = dataset.batch(60000) # drop_remainder is important on TPU, batch size must be fixed
    #dataset = dataset.prefetch(-1)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
    return dataset      

'''
Loads and maps the validation split of the dataset using the map function. Note that we try to load the gcs version since TPU can only work with datasets on Google Cloud Storage.
'''  
def get_validation_dataset():
    dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)

    #dataset = dataset.cache() # this small dataset can be entirely cached in RAM
    #dataset = dataset.batch(BATCH_SIZE) # 10000 items in eval dataset, all in one batch
    #dataset = dataset.prefetch(-1)
    return dataset

# instantiate the datasets

def get_bbox_mnist_dataset():
    training_dataset = get_training_dataset()
    validation_dataset = get_validation_dataset()

    images = []
    labels = []
    for img, l in validation_dataset:
        images.append(img.numpy())
        labels.append(np.hstack((l[0].numpy().argmax(), l[1].numpy())))
    X_test = np.array(images)
    labels = np.array(labels)
    y_test = (labels[:,0], labels[:,1:])


    images = []
    labels = []
    for img, l in training_dataset:
        images.append(img.numpy())
        labels.append(np.hstack((l[0].numpy().argmax(), l[1].numpy())))
    X_train = np.array(images)
    labels = np.array(labels)
    y_train = (labels[:,0], labels[:,1:])

    return (X_train, y_train), (X_test, y_test)
