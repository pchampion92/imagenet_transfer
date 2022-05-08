import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

from utils import visualize_sample, data_augmenter_builder, model_builder, plot_score
from constants import DATA_DIRECTORY, BATCH_SIZE, IMG_SIZE, IMG_SHAPE,LEARNING_RATE,NB_EPOCHS


def run():
    train_dataset = image_dataset_from_directory(DATA_DIRECTORY,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.2,
                                                 subset='training',
                                                 seed=42)
    validation_dataset = image_dataset_from_directory(DATA_DIRECTORY,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE,
                                                      validation_split=0.2,
                                                      subset='validation',
                                                      seed=42)
    if False:
        visualize_sample(train_dataset)

    # Prefetch data to optimize memory management. The buffer size (from which data is parsed) can be tuned automatically
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Preprocess input tool (normalization between [-1,1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Load the base model: do not include the top layer, as it will be edited for the classification
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model so it s does not change during training

    model = model_builder(IMG_SHAPE, preprocess_input, data_augmenter_builder(), base_model)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=NB_EPOCHS)

    plot_score(history)


if __name__ == '__main__':
    run()
