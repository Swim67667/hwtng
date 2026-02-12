import tensorflow as tf
import numpy as np
import tensorflow.keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

epoch=10
batch=64
lr=1e-4

(train_images, test_images), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
def preprocess_emnist(image, label):
    # EMNIST is natively (28, 28, 1)
    # 1. Normalize pixels to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # 2. Fix EMNIST's weird rotation/flip
    image = tf.transpose(image, perm=[1, 0, 2])
    # 3. EMNIST letters are 1-indexed (1-26). Subtract 1 to make it 0-indexed (0-25).
    return image, label - 1

# Prepare the data pipeline
train_images = train_images.map(preprocess_emnist).cache().shuffle(10000).batch(batch).prefetch(tf.data.AUTOTUNE)
test_images = test_images.map(preprocess_emnist).batch(batch).cache().prefetch(tf.data.AUTOTUNE)

"""
import_datapath = '/Users/alex/Documents/handwriting_data/dataset'
train_images = tensorflow.keras.preprocessing.image_dataset_from_directory(
    os.path.join(import_datapath, 'train'),
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch,
    image_size=(32, 32),
    shuffle=True)
test_images = tensorflow.keras.preprocessing.image_dataset_from_directory(
    os.path.join(import_datapath, 'val'),
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch,
    image_size=(32, 32),
    shuffle=False)
"""
#train_images = train_images.reshape((-1, 32, 32, 1))
#test_images = test_images.reshape((-1, 32, 32, 1))
# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[np.argmax(train_labels[i])])
plt.show()
"""

model = tensorflow.keras.Sequential([
tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
tensorflow.keras.layers.MaxPooling2D((2, 2)),
tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tensorflow.keras.layers.MaxPooling2D((2, 2)),
tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tensorflow.keras.layers.Flatten(),
tensorflow.keras.layers.Dense(128, activation='relu'),
tensorflow.keras.layers.Dense(26)
])

model.summary()


model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, epochs=epoch, 
                    validation_data=(test_images))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  verbose=2)

print("acc:",test_acc*100, "% loss:",test_loss)
model.save('handwriting.keras')
print("Model saved to handwriting.keras")