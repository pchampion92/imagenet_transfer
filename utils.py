import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_sample(dataset):
    class_names = dataset.class_names
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def data_augmenter_builder():
    """
    Build a data augmentation pipeline
    :return:
    """
    data_augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2)
    ])
    return data_augmenter


def model_builder(image_shape, preprocessor, data_augmenter, base_model):
    inputs = tf.keras.Input(shape=image_shape)
    X = data_augmenter(inputs)
    X = preprocessor(X)
    Y = base_model(X, training=False)

    # Add layer for binary classification
    Y = tf.keras.layers.GlobalAveragePooling2D()(Y)
    Y = tf.keras.layers.Dropout(0.2)(Y)
    outputs = tf.keras.layers.Dense(units=1, activation='linear')(Y)
    model = tf.keras.Model(inputs, outputs)
    return model


def plot_score(history):
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
