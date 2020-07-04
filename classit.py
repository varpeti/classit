import tensorflow as tf
from PIL import Image as PImage
import numpy as np
import sys
import os
import warnings


def main(argv):
    if len(argv) < 3:
        print("Usage: python classit.py name path/img")
        sys.exit()

    name = argv[1]
    img = np.asarray(PImage.open(argv[2]))

    model = tf.keras.models.load_model(name + "/" + name)

    result = model(np.array([img])).numpy()[0]
    warnings.warn(str(result))
    return (np.where(result == max(result))[0][0])  # The index of the max value


if __name__ == "__main__":
    print( main(sys.argv) )
