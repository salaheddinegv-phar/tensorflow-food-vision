"""
train.py
--------
The canonical training pipeline for Food Vision 101.

Implements a 2-stage transfer learning strategy:
  1. Feature Extraction (frozen backbone)
  2. Fine-Tuning (last 10 layers only)

Design decisions:
  - Stage 3/4 (deep/unfrozen fine-tuning) intentionally omitted.
    Experiments showed severe overfitting beyond 10 unfrozen layers
    (train/val gap >10% with zero val improvement).
  - L2 regularization, dropout, and label smoothing applied throughout.
  - Early stopping with aggressive patience prevents wasted compute.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')
<<<<<<< HEAD
=======
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

## This for Memory Growth 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import create_data_loader
from src.model_utils import create_model, save_model, unfreeze_model
<<<<<<< HEAD
from src.evaluation import plot_top_confusions, plot_combined_history
=======
from src.evaluation import make_confusion_matrix, plot_combined_history
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1
from src.helper_functions import (
    create_tensorboard_callback,
    create_checkpoint_callback,
)

def train(args):
    # Mixed precision for faster training on modern GPUs
    mixed_precision.set_global_policy("mixed_float16")

<<<<<<< HEAD
    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
=======
    from tensorflow.keras import mixed_precision 
    mixed_precision.set_global_policy("mixed_float16")
    
    # ========== 1. Load Data ==========
    # FIXED: Added class_names as 4th return value
    train_data, test_data, num_classes, class_names = create_data_loader(
        args.train_dir, 
        args.test_dir, 
        batch_size=args.batch_size
    )
    print(f"Data loaded! {num_classes} classes")
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1

    # ========== 1. Data ==========
    train_data, test_data, num_classes, class_names = create_data_loader(
        args.train_dir,
        args.test_dir,
        batch_size=args.batch_size
    )
    print(f"[INFO] Loaded {num_classes} classes")

    # ========== 2. Model ==========
    model = create_model(num_classes=num_classes)
    model.summary()

    # ========== 3. Callbacks ==========
    tb_callback = create_tensorboard_callback(
        dir_name=args.log_dir,
        experiment_name=args.experiment_name,
    )
    ckpt_callback, best_chkpt_callback = create_checkpoint_callback(
        checkpoint_dir=args.checkpoint_dir
    )
<<<<<<< HEAD

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    )

    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        min_delta=0.001,
        verbose=1
=======
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,  # Use from args (default 5)
        restore_best_weights=True
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1
    )

    print("Callbacks created!")

    # ========== 4. Stage 1: Feature Extraction ==========
    print("\n" + "="*50)
    print("Stage 01: Feature Extraction")
    print("="*50)

    feature_history = model.fit(
        train_data,
        epochs=args.initial_epochs,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
<<<<<<< HEAD
        validation_steps=len(test_data),
        callbacks=[tb_callback, ckpt_callback, early_stop, lr_reducer, best_chkpt_callback]
    )

    # ========== 5. Stage 2: Fine-Tuning (10 layers) ==========
    print("\n" + "="*50)
    print("Stage 02: Fine-Tuning (last 10 layers)")
    print("="*50)

    unfreeze_model(model=model, num_layers=10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

=======
        validation_steps=len(test_data),  
        callbacks=[tb_callback, ckpt_callback, early_stop, best_chkpt_callback]
    )

    # ========== 5. Stage 2: Fine Tuning ==========
    print("Stage 02: Fine Tuning")
    unfreeze_model(model=model)
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1
    fine_history = model.fit(
        train_data,
        epochs=args.initial_epochs + args.fine_tune_epochs,
        initial_epoch=feature_history.epoch[-1],
        steps_per_epoch=len(train_data),
        validation_data=test_data,
<<<<<<< HEAD
        validation_steps=len(test_data),
        callbacks=[tb_callback, ckpt_callback, early_stop, lr_reducer, best_chkpt_callback]
=======
        validation_steps=len(test_data),  # FIXED: Full test set
        callbacks=[tb_callback, ckpt_callback, early_stop, best_chkpt_callback]
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1
    )

    # ========== 6. Evaluation ==========
    print("\n[INFO] Generating confusion matrix...")
    y_pred_probs = model.predict(test_data, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = []
    for _, labels in test_data:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true[:len(y_pred)])

    plot_top_confusions(
        y_true=y_true,
        y_pred=y_pred,
        classes=class_names,
        top_n=20,
        savefig=True,
        save_path="images/top_confusions.png"
    )

    # ========== 7. Save & Plot ==========
    save_model(model, filepath=args.model_save_path)

    plot_combined_history(
        feature_history,
        fine_history,
        savefig=True,
        save_path=args.save_training_curves
    )
    print(f"[INFO] Training curves saved to {args.save_training_curves}")

<<<<<<< HEAD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Food Vision 101 — Professional Training Pipeline"
    )
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--initial_epochs", type=int, default=50)
    parser.add_argument("--fine_tune_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (aggressive to prevent overfitting)")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="efficientnetv2b0_food101")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--model_save_path", type=str, default="models/full_food_classifier.keras")
    parser.add_argument("--save_training_curves", type=str, default="images/combined_training_curves.png")
=======
    if args.save_training_curves:
        print(f"Save loss curves into {args.save_training_curves} path")
        plot_combined_history(feature_history, fine_history, savefig=True, save_path=args.save_training_curves)
        print("loss curves saved!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Vision 101 Training")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test data")
    parser.add_argument("--initial_epochs", type=int, default=20)
    parser.add_argument("--fine_tune_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32, help="Adjust based on VRAM. Use 16 for RTX 3070 8GB")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--log_dir", type=str, default="Tensorflow_hub_log")
    parser.add_argument("--experiment_name", type=str, default="EfficientnetV2B0")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--model_save_path", type=str, default="models/food_classifier.keras")
    parser.add_argument("--confusion_matrix", action="store_true", default=True)
    parser.add_argument("--savefig", action="store_true", help="Save training images to images/folders")
    parser.add_argument("--save_training_curves", type=str, default="images/combined_training_curves.png",
                        help="Save training loss curves")
>>>>>>> d742bb3bb3aa1f7530abb0326c63b394d6fc01d1

    args = parser.parse_args()
    train(args)
