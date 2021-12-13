import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML


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


def fashion_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.98, stratify=y, random_state=42)
    X_unlaeled = X
    
    return X_unlaeled, (X_train, y_train), (X_test, y_test)



def cross_correlation_plot():
    # Primero configure la figura, el eje y el elemento de la trama que queremos animar
    fig, ax = plt.subplots(figsize=(20,8) )
    plt.close()

    y1 = np.array([0.1,0.2,-0.1,4.1,-2,1.5,-0.1])
    x1=np.arange(1,len(y1)+1)
    ax.plot(x1,y1+7,'o-')

    ax.set_xlim(( -7, 15))
    ax.set_ylim((-3, 12))
    ax.set_yticks([])
    ax.set_xticks([])
    line, = ax.plot([], [], 'o-r')


    # función de inicialización: traza el fondo de cada cuadro
    def init():
        line.set_data([], [])
        return (line,)

    # función de animación
    def animate(i):
        i=i-6
        y2 = np.array([0.1,4,-2.2,1.6,0.1,0.1,0.2])
        x=np.arange(1,len(y2)+1)+i
        line.set_data(x, y2)
        ax.set_title('cross correlation=%.3f' %(np.correlate(y1,y2,mode='full'))[6+i], fontsize=20)
        ax.set_xlabel(f"{np.correlate(y1,y2,mode='full')[:7+i]}", fontsize=20)
        for t in ax.texts:
            t.set_visible(False)

        for i in range(len(x)):
            ax.text(x1[i], y1[i]+7, str(y1[i]), fontsize=20)
            ax.text(x[i], y2[i], str(y2[i]), fontsize=20)

            
        return (line,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=7+6, interval=2000, blit=True)

    rc('animation', html='jshtml')
    return anim
