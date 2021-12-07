#!/usr/bin/env python

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

test_df=pd.read_csv('data\X_test.csv')
test_df_label=pd.read_csv('data\y_test.csv')

num_test_examples = len(test_df)
dataset_test = tf.data.Dataset.from_tensor_slices((test_df.values.reshape(-1, 32, 32,1), test_df_label.to_numpy()))

batch_size = 64
testing_batches = dataset_test.shuffle(num_test_examples//4).batch(batch_size).map(normalize).prefetch(1)


arabic_characters = ['0','alef أ', 'beh ب', 'teh ت', 'theh ث', 'jeem ج', 'hah ح', 'khah خ', 'dal د', 'thal ذ',
                    'reh ر', 'zain ز', 'seen س', 'sheen ش', 'sad ص', 'dad ض', 'tah ط', 'zah ظ', 'ain ع',
                    'ghain غ', 'feh ف', 'qaf ق', 'kaf ك', 'lam ل', 'meem م', 'noon ن', 'heh هـ', 'waw و', 'yeh ي','hamza ء']



filepath='test_model5.h5'
my_model=tf.keras.models.load_model(filepath)

print(my_model.summary())

plt.figure(figsize=(10,10))
i=0
for image_batch, label_batch in testing_batches.take(24):
    ps = my_model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]
    i+=1
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(first_image, cmap = plt.cm.binary)
    plt.xlabel(arabic_characters[np.argmax(ps[0])]  +  ' -> ' + arabic_characters[label_batch.numpy().squeeze()[0]])
plt.show()


for image_batch, label_batch in testing_batches.take(1):
    ps = my_model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]
    
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(first_image, cmap = plt.cm.binary)
    ax1.axis('off')
    ax2.barh(np.arange(30), ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(30))
    ax2.set_yticklabels(arabic_characters)
    ax2.set_title('Class Probability')
    plt.title(arabic_characters[np.argmax(ps[0])]  +  ' -> ' + arabic_characters[label_batch.numpy().squeeze()[0]])
    ax2.set_xlim(0, 1.1)
    #plt.tight_layout()

plt.show()


for image_batch, label_batch in testing_batches.take(1):
    loss, accuracy = my_model.evaluate(image_batch, label_batch)

print('\nLoss after training: {:,.3f}'.format(loss))
print('Accuracy after training: {:.3%}'.format(accuracy))

