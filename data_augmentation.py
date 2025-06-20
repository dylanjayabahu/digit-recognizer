import tensorflow as tf
import random
import numpy as np
import cv2
import math


##data augmentation functions
def random_brightness(image, brightness_range):
    seed = (random.randint(0, 10), random.randint(0, 10))
    return tf.image.stateless_random_brightness(image, brightness_range, seed)

def random_contrast(image, lower_bound, upper_bound):
    seed = (random.randint(0, 10), random.randint(0, 10))
    return tf.image.stateless_random_contrast(image, lower_bound, upper_bound, seed)

def random_crop(image, min_pad, max_pad):
    padding_width = random.randint(min_pad, max_pad)
    seed = (random.randint(0, 10), random.randint(0, 10))
    padding = ( (padding_width, padding_width), (padding_width, padding_width), (0, 0)   )
    padded_image = tf.pad(image, padding, mode='constant')

    return tf.image.stateless_random_crop(padded_image, (28, 28, 1), seed)

##augment full dataset function
def dataset_with_augmentation(x_data, y_data, transformations, batch_size):
    mem_ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    mem_ds = mem_ds.shuffle(100)

    # get random generator for transformations
    # rng1 = tf.random.Generator.from_seed(123, alg='philox')
    
    for transform in transformations:
        # crop 
        if 'crop' == transform["name"]:
            min = transform["min_pad"]
            max = transform["max_pad"]
            mem_ds = mem_ds.map(lambda x, y: (random_crop(x, min, max), y), num_parallel_calls=tf.data.AUTOTUNE)

        # brightness
        if 'brightness' == transform["name"]:
            r = transform['range']
            mem_ds = mem_ds.map(lambda x, y: (random_brightness(x,r), y), num_parallel_calls=tf.data.AUTOTUNE)

        # contrast
        if 'contrast' == transform:
            min = transform["min"]
            max = transform["max"]
            mem_ds = mem_ds.map(lambda x, y: (random_contrast(x, min, max), y), num_parallel_calls=tf.data.AUTOTUNE)

    mem_ds = mem_ds.batch(batch_size)
    mem_ds = mem_ds.prefetch(tf.data.AUTOTUNE)
    
    return mem_ds

def view_augmentation():
    ##cycles through the dataset to view how the images look after being augmented
    ##called when this file is run directly, but not when it is imported to other files

    ##get mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train_all, y_train_all), (_, _) = mnist.load_data()
    x_train_all = np.expand_dims(x_train_all, axis=-1)


    ##use the default transformations
    transformations = [
            {"name":"crop",
            "min_pad":0,
            "max_pad":5},

            {"name":"brightness",
            "range":0.07},

            {"name":"contrast",
            "min":0.85,
            "max":1.3}
            ]

    ##augment the data
    train_ds = dataset_with_augmentation(x_train_all, y_train_all, transformations=transformations, batch_size=len(x_train_all))

    ##get the images and labels from the augmented dataset
    for thing in train_ds:
        images=thing[0]
        labels=thing[1]
    

    #iterate through the dataset and show the image while printing the label
    n=0
    while True:
        image = images[n]
        label = labels[n]

        print("Label: " + str(np.array(label)), end="\r")
        cv2.imshow("Image", np.array(tf.reshape(image, shape=(28,28))))

        ##if the user presses any of these keys, show the previous image
        if cv2.waitKey(0) in [ord("b"), ord("<"), ord(","), ord("-")]:
            n-=1

        ##otherwise, show the next image when a key is pressed
        else: 
            n+=1
    

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   view_augmentation()