import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os


def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def loadImages(path, imageName=None):
    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        if imageName is None or imageName == image:  # load only the [name] image or all
            loadedImages.append(np.asarray(Image.open(path + image)))
    return loadedImages


def loadAllImages(name):
    trainImages = []
    trainLabels = []

    for i in os.listdir(name + "/classes"):
        imgs = loadImages(name + "/classes/" + i + "/")

        if len(imgs) < 1:
            continue

        if len(trainImages) > 0:
            trainImages = np.concatenate([trainImages, imgs])
            trainLabels = np.concatenate([trainLabels, np.array([int(i)] * len(imgs))])
        else:
            trainImages = np.array(imgs)
            trainLabels = np.array([int(i)] * len(imgs))
    return trainImages, trainLabels


def loadThisImage(name, imageName):
    trainImages = []
    trainLabels = []

    for i in os.listdir(name + "/classes"):
        imgs = loadImages(name + "/classes/" + i + "/", imageName)

        if len(imgs) < 1:
            continue

        trainImages = np.array(imgs)
        trainLabels = np.array([int(i)] * len(imgs))
        break  # We only load the 1st matching name

    return trainImages, trainLabels


def main(argv):
    if len(argv) < 2:
        print("Usage: python train.py name [iteration]")
        print("\titeration:")
        print("\t\t[number]: how many times train on all images")
        print("\t\t[nothing]: train 1 times on all images ")
        print("\t\t[image_name]: train only on this [image_name], attempts to find the 1st matching occurrence")
        sys.exit()

    name = argv[1]
    iteration = 1
    trainImages, trainLabels = [], []
    if len(argv) >= 3:
        if isInt(argv[2]):
            iteration = int(argv[2])
            trainImages, trainLabels = loadAllImages(name)
        else:
            trainImages, trainLabels = loadThisImage(name, argv[2])
    else:
        trainImages, trainLabels = loadAllImages(name)

    model = tf.keras.models.load_model(name + "/" + name)

    print(len(trainImages), iteration)

    if len(trainImages) < 1:
        return

    for i in range(iteration):
        model.train_on_batch(trainImages, trainLabels)
        loss, acc = model.evaluate(trainImages, trainLabels, verbose=1)

    model.save(name + "/" + name)


if __name__ == "__main__":
    main(sys.argv)
