import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import os

if (len(sys.argv) < 5):
    print("Usage: python create.py name numberOfClasses width height")
    sys.exit()

name = sys.argv[1]
numOfClass = int(sys.argv[2])
width = int(sys.argv[3])
height = int(sys.argv[4])

os.mkdir(name)
os.mkdir(name+"/classes")
for i in range(0,numOfClass):
    os.mkdir(name+"/classes/"+str(i))

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(numOfClass))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.save(name+"/"+name)