import tensorflow as tf
from tensorflow import keras
import numpy as np
import gi
import threading
import time

gi.require_version("Gtk", "3.0")  # Use GTK 3.0
from gi.repository import Gtk, Gdk, GdkPixbuf
import cairo

# Load the trained model
try:
    model = keras.models.load_model("mnist_cnn.keras")
    print("Model loaded from mnist_cnn.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# GTK 3 Application
class DigitRecognizerApp(Gtk.Application):
    def __init__(self):
        super(DigitRecognizerApp, self).__init__()
        self.surface = None
        self.predict_timer = None
        self.last_draw_time = 0

    def do_activate(self):
        self.window = Gtk.Window(type=Gtk.WindowType.TOPLEVEL)
        self.window.set_title("Digit Recognizer")
        self.window.set_default_size(400, 400)
        self.window.connect("destroy", Gtk.main_quit)

        self.box = Gtk.VBox(spacing=10)
        self.window.add(self.box)

        # Drawing area
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(280, 280)
        self.drawing_area.connect("draw", self.on_draw)
        self.box.pack_start(self.drawing_area, True, True, 0)

        # Clear button
        self.clear_button = Gtk.Button(label="Clear")
        self.clear_button.connect("clicked", self.on_clear_clicked)
        self.box.pack_start(self.clear_button, False, False, 0)

        # Prediction label
        self.label = Gtk.Label(label="Predicted Digit: ")
        self.box.pack_start(self.label, False, False, 0)

        # Enable drawing events
        self.drawing_area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_MOTION_MASK
        )
        self.drawing_area.connect("motion-notify-event", self.on_motion_notify)
        self.drawing_area.connect("button-press-event", self.on_button_press)

        self.window.show_all()

    def on_draw(self, area, cr):
        if self.surface:
            cr.set_source_surface(self.surface, 0, 0)
            cr.paint()
        else:
            cr.set_source_rgb(1, 1, 1)  # White background
            cr.rectangle(0, 0, area.get_allocated_width(), area.get_allocated_height())
            cr.fill()

    def on_button_press(self, widget, event):
        if event.button == 1:  # Left mouse button
            if self.surface is None:
                self.surface = cairo.ImageSurface(cairo.Format.ARGB32, 280, 280)
                self.drawing = cairo.Context(self.surface)
                self.drawing.set_source_rgb(1, 1, 1)  # White background
                self.drawing.paint()
                self.drawing.set_source_rgb(0, 0, 0)  # Black for drawing
                self.drawing.set_line_width(20)
                self.drawing.set_line_cap(cairo.LineCap.ROUND)
            self.drawing.move_to(event.x, event.y)

    def on_motion_notify(self, widget, event):
        if event.state & Gdk.ModifierType.BUTTON1_MASK:
            self.drawing.line_to(event.x, event.y)
            self.drawing.stroke()
            self.drawing_area.queue_draw()

            current_time = time.time()
            if current_time - self.last_draw_time > 0.05:  # Throttle predictions
                self.last_draw_time = current_time
                self.start_predict_timer()

    def start_predict_timer(self):
        if self.predict_timer is not None:
            self.predict_timer.cancel()

        self.predict_timer = threading.Timer(0.5, self.predict_after_delay)
        self.predict_timer.start()

    def predict_after_delay(self):
        self.predict_timer = None
        self.predict_digit()

    def on_clear_clicked(self, button):
        self.surface = None
        self.drawing_area.queue_draw()
        self.label.set_label("Predicted Digit: ")

    def predict_digit(self):
        if self.surface:
            pixbuf = Gdk.pixbuf_get_from_surface(self.surface, 0, 0, 280, 280)
            pixbuf = pixbuf.scale_simple(28, 28, GdkPixbuf.InterpType.BILINEAR)

            data = pixbuf.get_pixels()
            image = np.frombuffer(data, dtype=np.uint8).reshape(28, 28, -1)
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Grayscale
            image = image.reshape(28, 28, 1).astype("float32") / 255.0
            image = np.expand_dims(image, axis=0)

            predictions = model.predict(image)
            predicted_digit = np.argmax(predictions)
            self.label.set_label(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":  # This is crucial
    app = DigitRecognizerApp()
    app.run(None)
    Gtk.main()
