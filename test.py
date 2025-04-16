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
import tensorflow as tf
from astropy.io import fits
from model import build_model, mix_loss, ssim_metric

# Load and preprocess data
def load_data(path):
    data_row = [(name, fits.open(os.path.join(path, name))[0].data) for name in os.listdir(path)]
    data_sort = sorted(data_row, key=lambda x: x[0])
    data = [item[1] for item in data_sort]
    name = [item[0] for item in data_sort]
    return tf.Variable(initial_value=data, dtype=tf.float32), name

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'mix_loss': mix_loss, 'ssim_metric': ssim_metric})
    return model

def predict(model, input_data):
    return model.predict(input_data)

if __name__ == "__main__":
    model = load_model("model/model_solarcnn")
    # Load test data and predict
    test_input, test_name = load_data("data/image/test/")
    output_path = "data/pred/"
    for i in range(test_input.shape[0]):
        print("Predict sample "+str(i+1))
        test_sample = tf.reshape(test_input[i], (-1, 256, 256, 1))
        prediction = predict(model, test_sample)
        pred_name = "enh_mdi_"+test_name[i][4:20]+"SolarCNN.fits"
        fits_file = os.path.join(output_path+pred_name)
        fits_data = fits.PrimaryHDU(prediction[0])
        fits_data.writeto(fits_file, overwrite=True)
    print("Finished")
