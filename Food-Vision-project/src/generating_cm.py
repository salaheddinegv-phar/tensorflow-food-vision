# generate_cm.py
import numpy as np
import tensorflow as tf
from data_preparation import create_data_loader
from evaluation import  plot_top_confusions

# 1. Configuration (Match these to your train.py args)
TRAIN_DIR = "data/train"  # or whatever your path is
TEST_DIR = "data/test"    # path to your validation/test set
BATCH_SIZE = 32

print("🔄 Loading data loaders...")
# We only need test_data and class_names from this
_, test_data, _, class_names = create_data_loader(
    TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE
)

print("🔄 Loading your trained model...")
model = tf.keras.models.load_model("models/food_classifier.keras")

print("🔮 Generating predictions and extracting labels in one pass...")

# 1. Initialize empty lists to collect data
all_labels = []

# 2. Loop through the dataset exactly ONCE
for images, labels in test_data:
    all_labels.append(labels.numpy())

# 3. Collapse the true labels into a single numpy array
y_true = np.concatenate(all_labels, axis=0)
test_labels = np.argmax(y_true, axis=1)

# 4. Now generate predictions (this will only take 198 steps now!)
print("🧠 Computing model predictions...")
y_pred_probs = model.predict(test_data)
y_preds = np.argmax(y_pred_probs, axis=1)

print("🎨 Generating confusion matrix...")
# Call your fixed function
plot_top_confusions(
    y_true=test_labels, 
    y_pred=y_preds, 
    classes=class_names,
    top_n = 20,
    figsize=(14,10),
    savefig=True     # Saves it to your images folder
)