# Digit_Recognizer
Machine Learning project using tensorflow to train a convolutional neural network on the MNIST dataset of digits. Code to create and train the model found in train_model.py and data_augmentation.py.
Using external webcam, user can draw a digit on sticky note/piece of paper on a black background. Code will locate the rectangle paper and isolate the digit, then uses tf/tflite model to predict what digit is shown. Drawing code is found in physical_draw.py.

![exmpe](https://github.com/dylanj1383/Digit_Recognizer/assets/109835004/7554589d-0730-4ec1-ba69-06b35bf710fd)

### Prerequisites
- Required Python Packages: OpenCV, Numpy
  - To run training code, tensorflow package is required
  - To only run drawing code, only tensorflow lite (tflite) package is requried. With tflite, project can even be run on a raspberry pi. 
 
### Usage
Run physical_draw.py from text editor, or run from command line:
```
python physical_draw.py
```
Place sticky note/paper in camera view, with a dark background. If a digit is recognized on the paper, the model prediction will be shown on the live camera feed.
Orientation of the sticky note shouldn't matter as it should be handled by the code. 
Press 'q' to quit the program. 
