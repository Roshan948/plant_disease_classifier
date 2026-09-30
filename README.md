# Tomato Leaf Disease Classification

ML course project. The model looks at a photo of a tomato leaf and tells which disease it has, or if the leaf is healthy. I compared a CNN I built myself with transfer learning and made a small app to try it.

## What's in the folder

- plant_disease_classifier.ipynb : the notebook, run it in Colab with a GPU
- plant_app.py : tkinter app for running it locally
- plant_disease_model.keras : saved best model
- class_names.json : class names the app needs

## Dataset

- New Plant Diseases Dataset from Kaggle (uploaded by vipoooool)
- About 87K leaf images in 38 classes (crop + disease), one folder per class
- Already split 80/20 into train and valid folders
- It was made by augmenting the original PlantVillage images offline
- I only used the tomato classes (10 of the 38), the full set is too heavy for Colab
- Images are copied into a separate tomato folder so the keras loader can read only those

## Problem

- Multi class image classification, leaf photo in, disease name out
- Goal was to compare a custom CNN against transfer learning and see how much augmentation matters
- End result is an app where you pick a leaf image and get the top 3 predictions with confidence

## Preprocessing

- Checked the class counts and the ratio between biggest and smallest class
- Looked at one sample image per class and checked image sizes
- Checked every file opens properly (corrupted files would be deleted) and looked at the file extensions
- Resized all images to 128x128, batch size 32
- Split the valid folder in half with a fixed seed: one half for validation while training, other half kept as the test set
- Pixel scaling is inside the models: divide by 255 for the custom CNN, scale to -1..1 for MobileNetV2
- Augmentation (random layers, only active while training):
  - flip (horizontal and vertical)
  - rotation (0.1)
  - zoom (0.1)
  - contrast (0.1)

## Why I did it this way

- Tomato subset only, because the whole dataset was too big for Colab
- 128x128 instead of the original size to keep training time down
- Augmentation as layers inside the model so it is switched off automatically when predicting
- Scaling inside the model so the saved file works on raw images and the app doesn't need extra steps
- MobileNetV2 because it is small and fast, and pretrained weights help when training time is limited
- Frozen first, then unfroze the last 30 layers with a tiny learning rate (1e-5) so the pretrained features don't get destroyed
- Early stopping on validation loss (patience 3) so the models don't overfit
- Picked the final model using validation accuracy, not test accuracy, so the test set is only for reporting
- Saved every run to disk because the fine tuning step reuses the same model object
- tkinter app run locally, not in Colab

## Experiments

- Split: train folder for training, valid folder split in half (validation and test)
- Adam optimizer, sparse categorical crossentropy, accuracy as the metric
- 4 runs:
  - custom CNN without augmentation (up to 15 epochs, lr 1e-3)
  - custom CNN with augmentation (up to 15 epochs, lr 1e-3)
  - MobileNetV2 frozen (up to 8 epochs, lr 1e-3)
  - MobileNetV2 fine tuned (up to 6 epochs, lr 1e-5)
- Custom CNN: 3 conv + max pool blocks (32, 64, 128 filters), global average pooling, dense 128, dropout 0.3
- Compared val accuracy, test accuracy, number of parameters and training time
- Looked at training curves, classification report, confusion matrix and some sample predictions

## Results

Fill these in from the notebook output.

- Custom CNN without augmentation: val ___ / test ___
- Custom CNN with augmentation: val ___ / test ___
- MobileNetV2 frozen: val ___ / test ___
- MobileNetV2 fine tuned: val ___ / test ___
- Best model: ___
- Augmentation effect on the custom CNN: ___
- Transfer learning vs custom CNN: ___ (training time: ___)
- Classes that got confused the most (from the confusion matrix): ___
- Leaf ID overlap between train and valid (from the leakage check): ___

## Limitations

- The dataset was augmented before it was split, so copies of the same leaf can be in both train and valid. Accuracy may look better than it really is
- Only tomato classes
- PlantVillage photos are clean leaves on plain backgrounds, so photos from a real field could do worse
- Test set is half of the valid folder, not a separate set

## Running the app

- Keep plant_app.py, plant_disease_model.keras and class_names.json in the same folder
- pip install tensorflow pillow numpy
- python plant_app.py
- Install the same tensorflow version that Colab had or the model might not load
