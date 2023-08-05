import tensorflow as tf
import cv2
import numpy as np
import os

## change these directories to match your system
TENSORBOARD_DIR = '/home/intern/code/Digit_Recognizer/tensorboard_logs/'
SAVED_MODELS_DIR = '/home/intern/code/Digit_Recognizer/saved_models/'


def make_model(show_summary = True):
    ##make model

    model=tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(6,(3,3),input_shape=(28,28,1),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(10,(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Flatten(input_shape=(28,28))) 

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']) 
    
    ##show summary if requested
    if show_summary:
        model.summary()

    return model


def train_model(model, epochs=1, early_stopper_patience = 20):


    ##early stopper stops training if validation loss stops decreasing
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                     min_delta=0, 
                                                     patience=early_stopper_patience, 
                                                     verbose=1,
                                                     restore_best_weights=True)
    
    ##save data to tensorboard for visualization
    model_filename = input("Do you want to save metrics to tensorboard? Enter model name here: ")

    if model_filename != "":
        path = os.path.join(TENSORBOARD_DIR, model_filename+"_tensorboard_data") 

        assert not os.path.exists(path)

        os.makedirs(path) 
        log_dir = TENSORBOARD_DIR+ model_filename+"_tensorboard_data"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                            histogram_freq=1, 
                                                            write_graph=True,
                                                            # update_freq='batch'
                                                            )

    #fit model
    model.fit(x_train, y_train,
            epochs=epochs, 
            verbose = 1,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stopper, tensorboard_callback])

    return model, model_filename


def save_model(model, name = ""):
    if name != "":    
        model.save(SAVED_MODELS_DIR + name)
        print("Model saved under filename:", name)
    else:
        print("No filename entered. Not saving model.")


def test_model(x_data, y_data, model, prompt_user=True):
    if prompt_user:
        command = input("Press enter to test this model. Input anything else to skip testing. ")
    else:
        command = ''
    
    
    if command == '':
        results = model.evaluate(x_data, y_data)
        print("testing loss:", round(results[0], 5), " "*6+"testing accuracy:", round(results[1], 5))
        return round(results[0], 5), round(results[1], 4)


def load_model(load_model_name):
    try:
        model = tf.keras.models.load_model(SAVED_MODELS_DIR + load_model_name)
        model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']) 
        return model
    

    except:
        ##if user has inputted an invalid filename, ask them to try again
        print("Invalid filename. Enter a valid model saved in " + SAVED_MODELS_DIR)
        return load_model(input("Input name for model: "))


def unisonShuffleDataset(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_data(percentage = 1, validation_split = 0.15):
    #load data 
    mnist = tf.keras.datasets.mnist
    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

    x_train_all, y_train_all = unisonShuffleDataset(x_train_all, y_train_all)

    ##discard some training data based on percentage
    x_train_all = x_train_all[:int(len(x_train_all)*percentage)]
    y_train_all = y_train_all[:int(len(y_train_all)*percentage)]


    #normalize data b/w 0 and 1
    x_train_all = x_train_all.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    #separate validation set
    num_validation_items = int(len(x_train_all)*validation_split)

    x_train = x_train_all[num_validation_items:]
    y_train = y_train_all[num_validation_items:]
    x_validation = x_train_all[:num_validation_items]
    y_validation = y_train_all[:num_validation_items]


    ##create correct shape for network
    x_train = np.expand_dims(x_train,axis=-1)
    x_validation = np.expand_dims(x_validation,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)

    x_train, y_train = unisonShuffleDataset(x_train, y_train)
    x_validation, y_validation = unisonShuffleDataset(x_validation, y_validation)
    x_test, y_test = unisonShuffleDataset(x_test, y_test)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_validation = tf.keras.utils.to_categorical(y_validation)
    y_test = tf.keras.utils.to_categorical(y_test)


    return x_train, y_train, x_validation, y_validation, x_test, y_test


def cycle_results(x_data, y_data, dimensions=(28, 28), model = ""):

    ##if no model has been provided, show the data and its labels only
    if model == "":
        output_data = np.array([])

    ##if a model has been provided, show the data, labels, and model predictions
    else:
        output_data = model.predict(x_data, verbose=0)


    ##cycle the results
    n=0

    while True:
        input_image = x_data[n].reshape(dimensions)
        title = "Label: " + str(np.argmax(y_data[n])) + " "


        if output_data.size > 0:
            title = title + "Prediction: " + str(np.argmax(output_data[n]))
            
        cv2.imshow("Image", input_image)

        print(title, end="\r")

        
        ##if "back" or any similar keys are pressed, show the previous example
        if cv2.waitKey(0) in [ord("b"), ord("<"), ord(","), ord("-")]:
            n-=1

        ##otherwise, show the next example when a key is pressed
        else: 
            n+=1



############################################################################################################
############################################################################################################
############################################################################################################

x_train, y_train, x_validation, y_validation, x_test, y_test = load_data()

command = input("Enter file name or press enter to train new network: ")


if command:
    model = load_model(command)

else:
    input("\nPress enter to continue with training model: ")

    model = make_model(show_summary=True)

    model, model_filename = train_model(model, epochs=1000, early_stopper_patience=20)

    save_model(model, name = model_filename)



test_model(x_test, y_test, model, prompt_user=True)

cycle_results(x_test, y_test, model=model)