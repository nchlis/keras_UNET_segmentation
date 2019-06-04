
# Keras implementation of a UNET for image segmentation

This repository provides a keras implementation of a [UNET](https://arxiv.org/abs/1505.04597) and tests it by performing segmentation on publicly available data.


Data: we will use the C. Elegans data available from the Broad Bioimage Benchmark Collection [here](http://localhost:8888/tree/Documents/PhD/Py_files/github_repos/keras_UNET_segmentation)

- [Input images](https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip): download and extract into ./data, then delete the '\_' within BBBC010_v2_images. Each input image corresponds to two input channels (Brightfield/GFP). Each channel corresponds to a separate grayscale image.
- [Target masks](https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip): download and extract into ./data


list of files:
- load_data.py: loads the data, splits into training, validation and test sets, and saves them to disk as numpy arrays. The images are resized to 400x400 pixels and normalized to [0,1].
- unet_train.py: trains a UNET implemented with keras on the training data. Saves the best model according to the performance on the validation set.
- unet_evaluate.py: evaluates the performance of the trained model on the left out test set.

