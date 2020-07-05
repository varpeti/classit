import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
import sys
import os


def roundUpPower2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


def main(argv):
    if len(argv) < 5:
        print("Usage: python create.py name numberOfClasses width height")
        sys.exit()

    name = argv[1]
    numOfClass = int(argv[2])
    width = int(argv[3])
    height = int(argv[4])
    awh = 32  # roundUpPower2()

    try:
        os.mkdir(name)
        os.mkdir(name + "/classes")
        for i in range(0, numOfClass):
            os.mkdir(name + "/classes/" + str(i))
    except:
        warnings.warn("Folders cannot be created!")

    model = models.Sequential()

    model.add(layers.Conv2D(awh, (3, 3), activation='relu', input_shape=(height, width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2 * awh, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2 * awh, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(awh, activation='relu'))
    model.add(layers.Dense(int(awh/2), activation='relu'))
    model.add(layers.Dense(numOfClass))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    model.save(name + "/" + name)


if __name__ == "__main__":
    main(sys.argv)
