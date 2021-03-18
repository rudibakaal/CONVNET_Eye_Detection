import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
import matplotlib.style as style
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = tf.keras.preprocessing.image.ImageDataGenerator()

data_dir = '/Users/rudibakaal/Downloads/eye_data'


datagen=ImageDataGenerator(rescale=1/255,validation_split=.2)

trainDatagen=datagen.flow_from_directory(
    data_dir,
    target_size=(100,100),
    batch_size=(32),
    class_mode='binary',
    subset='training',
    )


valDatagen=datagen.flow_from_directory(
    data_dir,
    target_size=(100,100),
    batch_size=(32),
    class_mode='binary',
    subset='validation'
)



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,kernel_size=(2,2),activation='relu',input_shape=(100, 100,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16,(2,2)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, (2, 2)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])


history = model.fit(trainDatagen,validation_data=valDatagen,epochs=10)

style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Binary Cross-entropy',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show()