import cv2
from train_model import load_model
import numpy as np
import time
from pprint import pprint
import random
import tensorflow as tf
import os

# HOME_DIR = '/home/intern/code/Digit_Recognizer/'
HOME_DIR = '/home/pi/code/Digit_Recognizer/'

def save_as_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16] ##make sure input is float16 not float32
    save_filename = '%s_float16.tflite' % "mnist_digit"
    tflite_model = converter.convert()
    
    print('Saving model as %s' % save_filename)
    
    with open(save_filename, 'wb') as f:
        f.write(tflite_model)

def predict_with_model(image):
    ##takes in a normalized 28x28 image and feeds it into model
    ##returns models prediction as an integer

    ##uses tflite model

    # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path="mnist_digit_float16.tflite")
    # interpreter.allocate_tensors()
    
    # # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    
    # # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(image, dtype=np.float16)
    # # assert input_data.shape == input_shape
    
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # interpreter.invoke()
    
    # output_data = interpreter.get_tensor(output_details[0]['index'])


    # return int(np.argmax(output_data[0])), 100*np.max(output_data[0])


    ##takes in a normalized 28x28 image and feeds it into model
    ##returns models prediction as an integer

    prediction = model.predict(np.array([image]), verbose=0)

    #print prediction and certainty
    print("Prediction: " + str(np.argmax(prediction[0])),  "Certainty: " + str(100*np.max(prediction[0])) + "%",  end="\r")


    return int(np.argmax(prediction[0])), 100*np.max(prediction[0])


def normalize_digit(image, visualize = True):

    ##make the image 28x28
    grayscale = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

    ##make it grayscale
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    ##subtract minimum value, then divide by maximum value
    # puts pixel vals between 0 and 1
    grayscale = grayscale-np.min(grayscale)
    grayscale = grayscale.astype(np.float32)
    grayscale /= np.max(grayscale)

    ##change invert from white bg to black txt --> black bb to white text (which model has been trained on)
    grayscale = 1-grayscale

    ##show the normalized image if visualize is True
    if visualize:
        cv2.imshow("grayscale ", grayscale)
        cv2.waitKey(1)


    return grayscale

def get_border_vals(img):
    image = np.array(img)
    assert image.ndim==2


    border_vals = []

    border_vals = border_vals + list(image[0]) + list(image[-1])
    image = np.transpose(image)

    border_vals = border_vals + list(image[0]) + list(image[-1])

    return border_vals

def crop_digit_from_frame(frame, visualize = True):

    original_frame = np.array(frame)

    cropped = None

    threshold = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

    tval,threshold = cv2.threshold(threshold, 0, 255, cv2.THRESH_OTSU)

    contours,hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours), "\n"*4)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.015 * peri, True)

        area = cv2.contourArea(contour)


        if len(approx) == 4 and area > 100:

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        
            # Calculate the rotation angle
            angle = rect[2]

            cv2.drawContours(frame, [box], 0, (180, 40, 0), 2)

            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

            # the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(original_frame, matrix, (width, height))

            if visualize:
                cv2.imshow("warped", warped)
                cv2.waitKey(1)

            warped_threshold = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, warped_threshold = cv2.threshold(warped_threshold, 0, 255, cv2.THRESH_OTSU)


            # print(warped_threshold.shape)

            ##remove the black outline edges by decreasing the size of the window and stopping once there are no black pixels in the border
            while True:
                if np.min(get_border_vals(warped_threshold)) != 0:
                    break
                
                warped_threshold = warped_threshold[1:warped_threshold.shape[0]-2, 1:warped_threshold.shape[1]-2]
                warped = warped[1:warped.shape[0]-2, 1:warped.shape[1]-2]


                if min(list(warped_threshold.shape)) <=2:
                    break

                # cv2.imshow("warped", warped)
                # cv2.waitKey(1)
                # cv2.imshow("warped_thresh", warped_threshold)
                # cv2.waitKey(-1)
            
            #there is a digit in the warped, cropped sticky note
            if np.min(warped_threshold.shape) > 2 and np.min(warped_threshold) == 0:

                highest = warped_threshold.shape
                lowest = (0, 0)
                leftest = warped_threshold.shape
                rightest = (0, 0)

                ##find the row of the highest, lowest, leftest, and rightest black pixel
                for r in range(len(warped_threshold)):
                    for c in range(len(warped_threshold[r])):

                        if warped_threshold[r][c] == 0:
                            if r < highest[0]:
                                highest = (r, c)
                            if r > lowest[0]:
                                lowest = (r, c)
                            if c > rightest[1]:
                                rightest = (r, c)
                            if c < leftest[1]:
                                leftest = (r, c)

                            # # cv2.circle(warped, center, 3, (155, 0, 0), 2)
                            # cv2.circle(warped, (100, 0), 3, (0, 0, 0), 2)
                            # cv2.circle(warped, (highest[1], highest[0]), 3, (0, 155, 0), 2)
                            # cv2.circle(warped, (lowest[1], lowest[0]), 3, (0, 0, 155), 2)
                            # cv2.circle(warped, (rightest[1], rightest[0]), 3, (255, 0, 0), 2)
                            # cv2.circle(warped, (leftest[1], leftest[0]), 3, (0, 0, 255), 2)

                            # cv2.imshow("warped", warped)
                            # cv2.waitKey(1)
                            # cv2.imshow("warped_thresh", warped_threshold)
                            # cv2.waitKey(-1)

                ##find the center of the digit
                center =  (int((leftest[1]+rightest[1])/2), int((highest[0] + lowest[0])/2))
                height = abs(highest[0]-lowest[0])
                width = abs(leftest[1]-rightest[1])

                # cv2.circle(warped, (highest[1], highest[0]), 3, (0, 155, 0), 2)
                # cv2.circle(warped, (lowest[1], lowest[0]), 3, (0, 0, 155), 2)
                # cv2.circle(warped, (rightest[1], rightest[0]), 3, (255, 0, 0), 2)
                # cv2.circle(warped, (leftest[1], leftest[0]), 3, (0, 0, 255), 2)
                # cv2.circle(warped, center, 3, (0, 0, 0), 2)

                square_length = max([height, width])
                ##############################################################################################################
                padding = int(round(square_length/4)) ##NEEED TO CHANGE THIS TO MAKE SURE IT DOESNT GO OFF OF THE CROPPED IMAGE>>>. Pad can only get so big, so we need to check that

                square_rad = (int(square_length/2) + padding) ##distance from the edges of swuare to center point

                y1 = center[1]- square_rad
                y2 = center[1]+ square_rad
                x1 = center[0]- square_rad
                x2 = center[0]+ square_rad

                # warped = cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 0, 0), 2)


                ##create a box around the digit 

                cropped = warped[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]



                if min([cropped.shape[0], cropped.shape[1]]) < 6:
                    cropped = None
                else:
                    if visualize:
                        cv2.imshow("warped", warped)
                        cv2.waitKey(1)
                        cv2.imshow("warped_thresh", warped_threshold)
                        cv2.waitKey(1)
                        cv2.imshow("final cropped", cropped)
                        cv2.waitKey(1)

                


    if visualize:
        cv2.imshow("threshold", threshold)
        cv2.waitKey(1)

    return original_frame, frame, cropped




##ensure SAVE_MODEL_DIR is set correctly in train_model.py and pass model name here. This will only be run once to save as tflite.
# if not os.path.exists(HOME_DIR + "mnist_digit_float16.tflite"):
#     model = load_model('mnist_classifier') 
#     save_as_tflite(model)

model = load_model('mnist_classifier')

col = (255, 0, 0) ##BGR Col for text on screen
cam = cv2.VideoCapture('/dev/video0')


cv2.namedWindow("Camera Output")


if not cam.isOpened():
    raise Exception('Could not open video device')

while True:
    success, frame = cam.read()
    # print(success, end="\r")

    if success:
        cam_window_size = frame.shape

        original_frame, frame, cropped = crop_digit_from_frame(frame, visualize=False)

        if cropped is not None:


            cropped_normalized = normalize_digit(cropped, visualize=False)

            cv2.imshow("Located Digit: ", cropped_normalized)


            prediction, certainty = predict_with_model(cropped_normalized)


        cv2.imshow("Camera Output", frame)
        key = cv2.waitKey(1) #wait 1 millisecond

    
    if key in [ord('q')]: ##q key pressed ==> quit
        break
