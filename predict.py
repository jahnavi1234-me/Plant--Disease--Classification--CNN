import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("models/cnn_model.h5")

IMG_SIZE = (224, 224)  # change ONLY if training used 128

img = image.load_img(r"C:\Users\DELL\OneDrive\Desktop\CNN_PROJECT\data\processed\test\Rice__brown_spot\IMG_20190420_185557.jpg", target_size=IMG_SIZE)
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print("Predicted class index:", np.argmax(pred))
