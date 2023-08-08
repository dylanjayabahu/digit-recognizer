import tkinter as tk
import math
import numpy as np
from train_model import load_model

def paint(event):
    global grid

    ##get the coordinates of the mouse
    x = event.x // 10
    y = event.y // 10

    ##get brush size and softness from sliders
    brush_softness = float(softness_slider.get())

    #paint in the range -brush size, brush size in the x and y direction
    for i in range(-brush_size, brush_size):
        for j in range(-brush_size, brush_size):

            ##only paint if the current pixel is within the radius of brush size
            if i**2 + j**2 <= brush_size**2:
                color = (1-brush_softness*math.sqrt(i**2 + j**2)/brush_size)*255
                if color > 0:
                  grid[(y+i)%28][(x+j)%28] = min(   (grid[(y+i)%28][(x+j)%28] +  color), 255    )
                grid[(y+i)%28][(x+j)%28] = max(0, grid[(y+i)%28][(x+j)%28])

                c =  grid[(y+i)%28][(x+j)%28]

                hexVal = "#%02x%02x%02x" % (int(c), int(c), int(c))
                screen.create_rectangle((x+j)*10, (y+i)*10, (x+j)*10+10, (y+i)*10+10, fill=hexVal, width = 0)

def update_brush_softness(val):
    #called whenever user drags bruhs softness slider
    softness_label.config(text="Brush softness: " + str(float(val)))

def reset_canvas():
    #called when user presses reset canvas button

    global grid

    #clear the screen and reset the grid
    screen.delete("all")
    for row in range(rows):
        for col in range(cols):
            grid[row][col] = 0
            screen.create_rectangle(col*10, row*10, col*10+10, row*10+10, fill='#000000')


def predict():
    ##called when user presses predict button

    ##normalize image for network
    image = np.expand_dims([grid], axis=-1)
    image = image/255

    ##use model to predict
    prediction = model.predict(np.array(image), verbose=0)
    
    #print prediction and certainty
    print("Prediction: " + str(np.argmax(prediction[0])),  "Certainty: " + str(100*np.max(prediction[0])) + "%",  end="\r")

def initialize():
    ##global the required variables that will be needed throughout the code
    global grid, screen, root, model, rows, cols
    global softness_slider, resetButton, predictButton
    global softness_label, brush_size

    ##load the classifier model
    model = load_model('mnist_classifier')

    ##size of the image (28x28)
    rows = 28
    cols = 28

    #initialize tkinter screen
    root = tk.Tk()

    screen = tk.Canvas(root, width=cols*10, height=rows*10, bg='#000000')
    screen.place(relx=.5, rely=.5, anchor='center')
    screen.pack()

    #create grid where user can draw their digit 
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            # 'pixel size' is 10
            x1 = col*10
            y1 = row*10
            x2 = x1+10
            y2 = y1+10
            color = grid[row][col]
            hexVal = "#%02x%02x%02x" % (int(color), int(color), int(color))
            screen.create_rectangle(x1, y1, x2, y2, fill=hexVal)

    ##bind whenever the user presses and moves mouse with paint function
    screen.bind('<B1-Motion>', paint)

    ##create the appropritae sliders and buttons 

    ##brush size
    brush_size=3

    ##softness slider changes brush softness
    softness_slider = tk.Scale(root, from_=1.5, to=2.5, resolution=0.05, orient="horizontal", command=update_brush_softness)
    softness_slider.pack()
    softness_label = tk.Label(root, text="Brush softness: 0.5\n\n")
    softness_label.pack()

    #reset button clears the grid and canvas
    resetButton= tk.Button(root, text ="Reset Canvas", command = reset_canvas)
    resetButton.pack()

    #predict button feeds the currently displayed canvas into the classifier mdoel
    predictButton = tk.Button(root, text ="Make Prediction", command = predict)
    predictButton.pack()


if __name__ == "__main__":
    initialize()
    root.mainloop()