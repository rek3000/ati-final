import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Recognizer")

        # Load the trained model
        self.model = keras.models.load_model("mnist_cnn.keras")

        # Create GUI components
        self.canvas = tk.Canvas(master, width=280, height=280, bg="white", cursor="cross")
        self.label = ttk.Label(master, text="Draw a digit", font=('Helvetica', 18))
        self.clear_btn = ttk.Button(master, text="Clear", command=self.clear_canvas)
        self.predict_btn = ttk.Button(master, text="Predict", command=self.predict_digit)

        # Layout components
        self.canvas.grid(row=0, column=0, pady=2, padx=2, columnspan=2)
        self.predict_btn.grid(row=1, column=0, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=1, pady=2, padx=2)
        self.label.grid(row=2, column=0, pady=2, padx=2, columnspan=2)

        # Set up drawing variables
        self.last_x = None
        self.last_y = None
        self.line_width = 15
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.line_width, fill="black", capstyle=tk.ROUND,
                smooth=True
            )
        self.last_x = event.x
        self.last_y = event.y

    def reset_position(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a digit")

    def predict_digit(self):
        # Save canvas as postscript image
        self.canvas.postscript(file="canvas.eps", colormode="mono")

        # Open and process the image
        img = Image.open("canvas.eps").convert('L')  # Convert to grayscale first
        img = ImageOps.invert(img)                   # Then invert colors
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        self.label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
