#!/usr/bin/env python3
"""Train a transfer learning model on CIFAR-10."""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for MobileNetV2."""
    X_p = K.applications.mobilenet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, Y_train), _ = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)

    inputs = K.Input(shape=(32, 32, 3))

    resized = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image, 3, 3, data_format="channels_last"
        )
    )(inputs)

    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(96, 96, 3)
    )

    base_model.trainable = False

    x = base_model(resized, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation="softmax")(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["acc"]
    )

    callback_list = [
        K.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=0.00001
        )
    ]

    model.fit(
        X_train,
        Y_train,
        validation_split=0.15,
        batch_size=128,
        epochs=25,
        callbacks=callback_list,
        verbose=1
    )

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["acc"]
    )

    model.fit(
        X_train,
        Y_train,
        validation_split=0.15,
        batch_size=128,
        epochs=10,
        callbacks=callback_list,
        verbose=1
    )

    model.save("cifar10.h5")
