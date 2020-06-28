import tensorflow as tf
from PIL import Image as PImage
import numpy as np
import sys
import os
import warnings

if (len(sys.argv) < 3):
    print("Usage: python classit.py name path/img")
    sys.exit()

name = sys.argv[1]
img = np.asarray(PImage.open(sys.argv[2]))

model = tf.keras.models.load_model(name + "/" + name)

result = model(np.array([img])).numpy()[0]
warnings.warn(str(result))
print(np.where(result == max(result))[0][0]) # The index of the max value
