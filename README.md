# Digit Recognizer App

This project implements a simple digit recognition application using TensorFlow/Keras for the machine learning model and Tkinter for the graphical user interface.  It allows users to draw digits on a canvas, and the application predicts the drawn digit using a pre-trained model.

## Getting Started

### 1. Prerequisites

*   Python 3.x (recommended 3.7 or higher)
*   `pip` (Python package installer)

### 2. Setting up the environment

1.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv  # Creates a virtual environment named "venv"
    ```

2.  **Activate the virtual environment:**

    *   **Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        venv\Scripts\activate
        ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 3. Training the model

1.  **Run the training script:**

    ```bash
    python train.py
    ```

    This script (`train.py`) will:

    *   Load the MNIST dataset.
    *   Define and train a Convolutional Neural Network (CNN) model.
    *   Save the trained model to a file named `mnist_cnn.keras`.

### 4. Running the application

1.  **Run the application script:**

    ```bash
    python app.py
    ```

    This will launch the Digit Recognizer application.

## How to use the application

1.  **Drawing:** Use the mouse to draw a digit on the white canvas. Click and drag to draw.
2.  **Prediction:** The application will predict the digit you've drawn and display the prediction and confidence level below the canvas.
3.  **Clearing:** Click the "Clear" button to erase the canvas and start drawing a new digit.

## Project Structure
```
Digit-Recognizer/
├── app.py        # The main application script (Tkinter GUI)
├── train.py      # The model training script
├── mnist_cnn.keras # The trained model file (saved by train.py)
├── venv/         # The virtual environment (if you created one)
└── README.md     # This file
```

## Model Details

The model used in this application is a simple CNN trained on the MNIST dataset.  It consists of convolutional layers, max pooling layers, a flatten layer, and fully connected layers.  The model is trained using the Adam optimizer and categorical cross-entropy loss.

## Further Improvements (Optional)

*   **GPU Acceleration:** If you have a compatible NVIDIA GPU, install the CUDA toolkit and the GPU version of TensorFlow (`tensorflow`) for faster training and prediction.
*   **Model Architecture:** Experiment with different CNN architectures to potentially improve accuracy.
*   **Data Augmentation:** Implement data augmentation during training to make the model more robust to variations in handwriting.
*   **User Interface Enhancements:** Add more features to the UI, such as different brush sizes, colors, or the ability to load and save images.
*   **Deployment:** Package the application for distribution to other users.

## Acknowledgements

*   This project uses the MNIST dataset, which is a widely used dataset for digit recognition.
*   The TensorFlow/Keras library is used for building and training the machine learning model.
*   The Tkinter library is used for creating the graphical user interface.
*   The Pillow and OpenCV libraries are used for image manipulation and processing.

