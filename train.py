import tensorflow as tf
from PIL import Image as PImage
import numpy as np
import sys
import os


def main(argv):
    if len(argv) < 2:
        print("Usage: python classit.py name")
        sys.exit()

    name = argv[1]
    model = tf.keras.models.load_model(name + "/" + name)

    def loadImages(path):
        imagesList = os.listdir(path)
        loadedImages = []
        for image in imagesList:
            img = PImage.open(path + image)
            loadedImages.append(np.asarray(img))
        return loadedImages

    trainImages = []
    trainLabels = []
    first = True

    for i in os.listdir(name + "/classes"):
        imgs = loadImages(name + "/classes/" + i + "/")

        if not first:
            trainImages = np.concatenate([trainImages, imgs])
            trainLabels = np.concatenate([trainLabels, np.array([int(i)] * len(imgs))])
        else:
            trainImages = imgs
            trainLabels = np.array([int(i)] * len(imgs))
            first = False

    num = 10
    while 1:
        model.train_on_batch(trainImages, trainLabels)
        loss, acc = model.evaluate(trainImages, trainLabels, verbose=1)
        num -= 1
        if acc > 0.9 or num < 1:
            break
    model.save(name + "/" + name)


if __name__ == "__main__":
    main(sys.argv)
