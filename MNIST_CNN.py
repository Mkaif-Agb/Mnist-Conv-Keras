import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from warnings import filterwarnings
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
filterwarnings('ignore')
training_path = 'Mnist/trainingSet'
test_path = 'Mnist/testSet'

model = Sequential()

model.add(Convolution2D(32,3,3, input_shape=(28,28,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('MNIST_Convolutional_Layer.h5', monitor='val_acc', verbose=1, save_best_only=True)
call_back_list = [checkpoint]

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Mnist/trainingSet',
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'Mnist/trainingSample',
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical')

history = model.fit_generator(
          training_set,
          samples_per_epoch=42000,
          epochs=10,
          callbacks=call_back_list,
          validation_data=test_set,
          validation_steps=600)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


from keras.models import load_model
import numpy as np
from keras.preprocessing import image
model = load_model('MNIST_Convolutional_Layer.h5')
prediction_image = image.load_img('Mnist/testSet/img_56.jpg', target_size=(28,28))
plt.imshow(prediction_image)
prediction_image = image.img_to_array(prediction_image)
prediction_image = np.expand_dims(prediction_image, axis=0)
prediction_image.shape
result = model.predict_classes(prediction_image)
# training_set.class_indices

if model.predict_classes(prediction_image) == 0:
    print('0')
elif model.predict_classes(prediction_image) == 1:
    print('1')
elif model.predict_classes(prediction_image) == 2:
    print("2")
elif model.predict_classes(prediction_image) == 3:
    print("3")
elif model.predict_classes(prediction_image) == 4:
    print("4")
elif model.predict_classes(prediction_image) == 5:
    print('5')
elif model.predict_classes(prediction_image) == 6:
    print("6")
elif model.predict_classes(prediction_image) == 7:
    print("7")
elif model.predict_classes(prediction_image) == 8:
    print("8")
elif model.predict_classes(prediction_image) == 9:
    print("9")


##########################################################

# Using Same Dataset from a folder with Keras

##########################################################

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)


train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

history = model.fit_generator(
                            train_generator,
                            samples_per_epoch=69600,
                            epochs=5,
                            validation_data=validation_generator,
                            validation_steps=17400)
