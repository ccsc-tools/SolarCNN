# =========================================================================
#   (c) Copyright 2025
#   All rights reserved
#   Programs written by Chunhui Xu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from astropy.io import fits
from model import build_model, mix_loss, ssim_metric

input_path = "data/image/train/"
label_path = "data/label/train/"
output_path = "model/"

# Load and preprocess data
def load_data(path):
    data_row = [(name, fits.open(os.path.join(path, name))[0].data) for name in os.listdir(path)]
    data_sort = sorted(data_row, key=lambda x: x[0])
    data = [item[1] for item in data_sort]
    return tf.Variable(initial_value=data, dtype=tf.float32)

if __name__ == "__main__":
    print("Data Loading")
    x_train = load_data(input_path)
    y_train = load_data(label_path)

    x_train = tf.clip_by_value(x_train, -2000., 2000.) / 2000.
    y_train = tf.clip_by_value(y_train, -2000., 2000.) / 2000.

    x_train = tf.reshape(x_train, (-1, 256, 256, 1))
    y_train = tf.reshape(y_train, (-1, 256, 256, 1))

    # Data augmentation
    x_train = tf.concat([x_train, tfa.image.rotate(x_train, 90)], 0)
    y_train = tf.concat([y_train, tfa.image.rotate(y_train, 90)], 0)

    x_train = tf.concat([x_train, tfa.image.rotate(x_train, 180)], 0)
    y_train = tf.concat([y_train, tfa.image.rotate(y_train, 180)], 0)
    print("Data Loaded")

    # Build and compile model
    model = build_model()
    model.compile(loss=mix_loss, optimizer=tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5), metrics=[ssim_metric])

    # Train model
    batch_size = 64
    epochs = 3000
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    # Save model
    model.save(os.path.join(output_path, "model_solarcnn_retrain"))