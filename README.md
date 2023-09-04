# Digit_Recognizer
Simple AI project using tensorflow to train a model on the MNIST dataset of digits. Using external webcam, user can draw a digit on sticky note/piece of paper on a black background. Code will locate the rectangle paper and isolate the digit, then uses tf/tflite model to predict what digit is shown. 

![exmpe](https://github.com/dylanj1383/Digit_Recognizer/assets/109835004/7554589d-0730-4ec1-ba69-06b35bf710fd)

### Prerequisites
- Required Python Packages: OpenCV, Numpy
  - To run training code, tensorflow package is required
  - To only run drawing code, tensorflow lite (tflite) package is requried
 
### Command
Run the physical_draw.py from command line or other text editor:
...
python physical_draw.py
...
