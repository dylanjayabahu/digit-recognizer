#ensure you have the required packages installed

tensorflow_package = 'tf' ##this should reflect wheather you have tensorflow installed or if you have just installed tflite_runtime
# cam_type = '/dev/video0'
cam_type = 0
if tensorflow_package not in ['tf', 'tflite']:
    raise Exception('Ensure you have installed either full tensorflow or just the tflite_runtime.\n Change the variable "tensorflow package" to reflect what you have installed.')


if tensorflow_package=="tf":
    import tensorflow as tf #use this if you have tensorflow installed
elif tensorflow_package=="tflite":
    import tflite_runtime.interpreter as tflite #for raspberry pi/if you don't have tf installed, use this

import cv2
# from train_model import load_model 
import numpy as np
import os


##change home_dir to reflect the file path on your system
# HOME_DIR = '/home/intern/code/Digit_Recognizer/'
# HOME_DIR = '/home/dylan/code/Digit_Recognizer/'
HOME_DIR = 'C:\\Users\\dylan\\OneDrive\\Desktop\\code\\Digit_Recognizer'

##only needed if you have trained your own model with tensorflow and want to save as tflite to run
# def save_as_tflite(model):
#     ##saves a regular tf model as tflite if not already done so
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_types = [tf.float16] 
#     save_filename = '%s_float16.tflite' % "mnist_digit"
#     tflite_model = converter.convert()
    
#     print('Saving model as %s' % save_filename)
    
#     with open(save_filename, 'wb') as f:
#         f.write(tflite_model)



def predict_with_model(image, print_result=True):
    ##called whenever a digit is recognized in the image


    ##this code uses tflite model
    image = np.array([image])
    image = np.expand_dims(image, axis = -1)
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])


    ##this code uses a tensorflow model that must be initalized somewhere in teh code
    # prediction = model.predict(np.array([image]), verbose=0)


    #print prediction and certainty
    if print_result:
        print("Prediction: " + str(np.argmax(prediction[0])),  "Certainty: " + str(100*np.max(prediction[0])) + "%",  end="\r")


    return int(np.argmax(prediction[0])), 100*np.max(prediction[0])


def normalize_digit(image, visualize = True):
    ##takes the cropped area where a digit is recognized and turns it into grayscale with black background for model

    ##make the image 28x28
    grayscale = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

    ##make it grayscale
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    ##subtract minimum value, then divide by maximum value
    # this puts pixel vals between 0 and 1
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
    #helper function that returns a list of only the border values in an MxN image

    #make sure there are only 2 dimensions to iterate through
    image = np.array(img)
    assert image.ndim==2

    #initalize the list of boder values
    border_vals = []

    ##get a list of the top and bottom values
    border_vals = border_vals + list(image[0]) + list(image[-1])

    ##transpose the image (flip it on its side) and then get a list of the new top and bottom (which is actually the left and right borders)
    image = np.transpose(image)
    border_vals = border_vals + list(image[0]) + list(image[-1])

    #return the result
    return border_vals

def crop_digit_from_frame(frame, visualize = True):
    #create a copy of the frame before we draw boxes and text on it
    original_frame = np.array(frame)

    ##initialize cropped to be none, so if we don't change it no prediction will be made
    cropped = None

    #create a threshold of the original image to locate the sticky note/paper
    threshold = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    tval,threshold = cv2.threshold(threshold, 0, 255, cv2.THRESH_OTSU)

    #find any contours from the thresholded image
    contours,hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##iterate through each contour and see if it is a rectangle of appropriate size
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
        area = cv2.contourArea(contour)


        if len(approx) == 4 and area > 100:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        
            # get the rotation angle
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

            ##create a new threshold of the sticky note, so we can locate the digit within the image
            warped_threshold = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, warped_threshold = cv2.threshold(warped_threshold, 0, 255, cv2.THRESH_OTSU)

            ##remove the black outline edges by decreasing the size of the window and stopping once there are no black pixels in the border. 
            ##this accounts for if the piece of paper is not a perfect square
            while True:

                ##exit if there are no more black pixels on the edges
                if np.min(get_border_vals(warped_threshold)) != 0:
                    break

                ##reduce the size of the rectangle by one if there are still black pixels on teh edge
                warped_threshold = warped_threshold[1:warped_threshold.shape[0]-2, 1:warped_threshold.shape[1]-2]
                warped = warped[1:warped.shape[0]-2, 1:warped.shape[1]-2]

                ##if the size of the rectangle gets too small, break since the contour found was not a proper rectangle
                if min(list(warped_threshold.shape)) <=2:
                    break

            
            #check if there is a digit in the warped, cropped sticky note
            if np.min(warped_threshold.shape) > 2 and np.min(warped_threshold) == 0:

                #find the center of the digit, by looking for the highest, lowest, leftest, and rightest black pixesl
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


                ##find the center of the digit uses the coordinates of the boundaries
                center =  (int((leftest[1]+rightest[1])/2), int((highest[0] + lowest[0])/2))
                height = abs(highest[0]-lowest[0])
                width = abs(leftest[1]-rightest[1])

                ##create an exact length for a square enclosing the digit
                square_length = max([height, width])

                ##add padding around the image as a quarter of the square length. This assumes that the user leaves some padding on the sticky note.
                padding = int(round(square_length/4))

                ##create a new square length with padding taken into account
                square_rad = (int(square_length/2) + padding) ##distance from the edges of swuare to center point

                ##define the coordinates of the cropped square
                y1 = center[1]- square_rad
                y2 = center[1]+ square_rad
                x1 = center[0]- square_rad
                x2 = center[0]+ square_rad

                ##crop the digit using these coords
                cropped = warped[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]


                ##make sure the square we have cropped is not too small (size larger than six)
                if min([cropped.shape[0], cropped.shape[1]]) < 6:
                    cropped = None
                
                ##otherwise, we can visalize the steps if needed
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

    #return the result
    return original_frame, frame, cropped

def run_main():
    ##main code to be run
    global model, input_details, output_details

    #if there is no tflite model that exists already, load a tf model and save it as tf lite
    #this is only needed if you have a different tf model saved
    # if not os.path.exists(HOME_DIR + "mnist_digit_float16.tflite"):
    #     model = load_model('mnist_classifier') 
    #     save_as_tflite(model)


    if tensorflow_package == "tf":
        ##load the tflite model
        model  = tf.lite.Interpreter("mnist_digit_float16.tflite") ##use this if you have tf installed
    
    elif tensorflow_package == "tflite":
        ##use the following code if you don't have tf isntalled (e.g. rpi)
        model = tflite.Interpreter("mnist_digit_float16.tflite")

    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()


    col = (255, 255, 255) ##BGR Col for prediction text on screen
    #get the video capture from a usb webcam

    cam = cv2.VideoCapture(cam_type) ##this may need to be changed 

    cv2.namedWindow("Camera Output") #create a window for camera output

    #ensure camera is working properly
    if not cam.isOpened():
        raise Exception('Could not open video device')

    ##run the main loop
    while True:
        success, frame = cam.read()

        if success:
            #locate the digit and crop 
            original_frame, frame, cropped = crop_digit_from_frame(frame, visualize=False)

            if cropped is not None:

                ##Normalize the digit so the network can recognize it (grayscae with black background)
                cropped_normalized = normalize_digit(cropped, visualize=False)

                ##show the located digit
                # cv2.imshow("Located Digit: ", cropped_normalized)

                ##predict with tflite model
                prediction, certainty = predict_with_model(cropped_normalized)

                ##show prediction on screen if certainty is above 60%
                if certainty > 60:
                    frame = cv2.putText(frame, "Prediction: " + str(prediction), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)



            #show the frame 
            cv2.imshow("Camera Output", frame)
            key = cv2.waitKey(1) #wait 1 millisecond before showing next frame

        
        if key in [ord('q')]: ##q key pressed ==> quit
            break


if __name__ == "__main__":
   ##excecute only when run not via import
   run_main()