# Food Vision 101 🍔🥗🥙🌮

**TensorFlow • Transfer Learning • 101 Food Classes**

**TensorFlow Transfer Learning Project** — Building a powerful image classifier that can recognize **101 different food types**.


## Overview

This project is part of my journey through **Daniel Bourke's TensorFlow Developer Certificate** course. It demonstrates end-to-end transfer learning using modern convolutional neural networks (EfficientNet) on the famous **Food101** dataset.

I started with **10% of the data** to get fast results, then scaled up to the **full dataset**. I also performed detailed error analysis using confusion matrices to understand where the model gets confused and how to improve it.

## Key Features

- Transfer Learning with **EfficientNetB0** (and B4)
- Data augmentation techniques
- Comprehensive model evaluation
- Confusion matrix analysis + error investigation
- Scaling from 10% to 100% of the dataset
- Clean, reproducible code structure

## Dataset

- **Name**: Food101
- **Classes**: 101 food categories
- **Training Images**: 75,750 (full) | 7,575 (10%)
- **Test Images**: 25,250
- Source: [Kaggle - Food101](https://www.kaggle.com/datasets/dansbecker/food-101)

## Results

### Model Performance

| Model                    | Dataset     | Accuracy | Precision | Recall | Training Time |
|--------------------------|-------------|----------|-----------|--------|---------------|
| EfficientNetB0           | 10%         | [XX.X%]  | [XX.X%]   | [XX.X%] | ~XX minutes   |
| EfficientNetB0           | Full        | [YY.Y%]  | [YY.Y%]   | [YY.Y%] | ~XX hours     |
| EfficientNetB4 (fine-tuned) | Full     | [ZZ.Z%]  | [ZZ.Z%]   | [ZZ.Z%] | ~XX hours     |

> *Add your actual results here after training*

### Visual Results

![Confusion Matrix](images/confusion_matrix.png)
![Prediction Examples](images/predictions_grid.png)

## Most Confused Classes

*(Example - update with your findings)*
- `apple_pie` ↔ `apple_strudel`
- `chicken_wings` ↔ `fried_chicken`
- `pancakes` ↔ `waffles`

I created custom code to analyze and address these confusions.

## Technologies Used

- **TensorFlow** 2.x
- **TensorFlow Hub**
- **EfficientNet** (pre-trained models)
- Matplotlib, Seaborn, Scikit-learn
- Google Colab

## Project Structure

```bash
food-vision-tensorflow/
├── notebooks/           # Main Jupyter notebooks
├── src/                 # Reusable Python modules
├── data/                # Dataset (ignored)
├── models/              # Saved models
├── images/              # Visualizations & results
├── README.md
├── requirements.txt
└── .gitignore
