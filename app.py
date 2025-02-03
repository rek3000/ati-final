import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from tensorflow import keras
import cv2


class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer 3000")
        master.geometry("350x400")  # Fixed window size

        # Styling
        master.configure(bg="#f0f0f0")  # Light gray background

        style = ttk.Style()
        style.theme_use("clam")  # Modern theme
        style.configure("TButton", padding=6, font=("Helvetica", 12), background="#e0e0e0") # Button style
        style.configure("TLabel", font=("Helvetica", 14), background="#f0f0f0") # Label style


        # Load the trained model
        self.model = keras.models.load_model("mnist_cnn.keras")

        # Create GUI components
        self.canvas = tk.Canvas(
            master, width=280, height=280, bg="white", cursor="cross"
        )
        self.label = ttk.Label(
            master, text="Draw a digit", font=("Helvetica", 16), anchor="center"
        )
        self.clear_btn = ttk.Button(master, text="Clear", command=self.clear_canvas)
        self.predict_btn = ttk.Button(
            master, text="Predict", command=self.predict_digit
        )

        # Layout configuration
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)

        self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=2)
        self.predict_btn.grid(row=1, column=0, pady=5, sticky="e")
        self.clear_btn.grid(row=1, column=1, pady=5, sticky="w")
        self.label.grid(row=2, column=0, pady=10, columnspan=2)

        # Drawing variables
        self.last_x = None
        self.last_y = None
        self.line_width = 10  # Increased line width for clearer strokes

        # Event bindings
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                width=self.line_width,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True,
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
        # Create a blank image
        img = Image.new("L", (280, 280), 255)  # White background
        draw = ImageDraw.Draw(img)

        # Draw all canvas lines onto the image
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:  # Only process lines
                draw.line(coords, fill=0, width=self.line_width)

        # Center and preprocess the image
        # img = self.center_image(img)
        # img = img.resize((28, 28))
        img = img.resize((28, 28), Image.LANCZOS)
        # image_array = 255 - np.array(img, dtype=np.float32)

        image_array = np.array(img, dtype=np.float32)

        # Apply adaptive thresholding
        image_array = cv2.adaptiveThreshold(
            image_array.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        ).astype(np.float32)

        # Normalize using MNIST parameters (same as training)
        image_array = (image_array / 255.0 - 0.1307) / 0.3081

        # image = image_array.reshape(-1, 28, 28, 1).astype(np.float32)
        image = image_array.reshape(1, 28, 28, 1).astype(np.float32)

        # Convert to model input format
        # img_array = np.array(img) / 255.0
        # img_array = img_array.reshape(-1, 28, 28, 1)

        # Get prediction
        prediction = self.model.predict(image, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Update label without window resize
        self.label.config(text=f"Prediction: {digit}\nConfidence: {confidence:.2%}")

    def center_image(self, img):
        """Center the digit in the image using bounding box"""
        # Convert to numpy array and find content bounding box
        img_data = np.array(img)
        rows = np.any(img_data < 255, axis=1)
        cols = np.any(img_data < 255, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Center the content
        bbox = (cmin, rmin, cmax, rmax)
        cropped = img.crop(bbox)
        new_img = Image.new("L", (28, 28), 255)
        new_img.paste(cropped, ((28 - (cmax - cmin)) // 2, (28 - (rmax - rmin)) // 2))
        return new_img


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

