
# Keras implementation of a UNET for image segmentation

list of files:
- load_data.py: loads the data, splits into training, validation and test sets, and saves them to disk as numpy arrays.
- unet_train.py: trains a UNET implemented with keras on the training data. Saves the best model according to the performance on the validation set.
- unet_evaluate.py: evaluates the performance of the trained model on the left out test set.

