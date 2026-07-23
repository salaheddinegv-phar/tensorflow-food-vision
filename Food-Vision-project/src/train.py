import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

## This for Memory Growth 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import argparse
import sys
from pathlib import Path

# Allow running from project root or src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.data_preparation import create_data_loader
from src.model_utils import create_model, save_model, unfreeze_model
from src.evaluation import make_confusion_matrix , plot_combined_history
from src.helper_functions import (
    create_tensorboard_callback,
    create_checkpoint_callback,
    combine_history,
)


def train(args):
    print(f"Train dir: {args.train_dir}")
    print(f"Test dir: {args.test_dir}")

    from tensorflow.keras import mixed_precision 
    mixed_precision.set_global_policy("mixed_float16") # in order to make training more faster with no accuracy loss
    # ========== 1. Load Data ==========
    
    train_data, test_data ,num_classes , class_namess = create_data_loader(args.train_dir, args.test_dir, batch_size=args.batch_size)
    print(f"Data loaded! {num_classes} classes")

    # ========== 2. Build Model ==========
    model = create_model(num_classes=num_classes)
    model.summary()

    # ========== 3. Callbacks ==========
    tb_callback = create_tensorboard_callback(
        dir_name=args.log_dir,
        experiment_name=args.experiment_name,
    )
    ckpt_callback , best_chkpt_callback= create_checkpoint_callback(
        checkpoint_dir=args.checkpoint_dir


    )
    print("Callbacks created!")

    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,           # stop after 3 epochs with no improvement
        restore_best_weights=True
    )

    # ========== 4. Stage 1: Feature Extraction ==========
    print("Stage 01: Feature Extraction")
    feature_history = model.fit(
        train_data,
        epochs=args.initial_epochs,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=int(0.25 * len(test_data)),
        callbacks=[tb_callback, ckpt_callback, early_stop, best_chkpt_callback]
    )


    # ========== 5. Stage 2: Fine Tuning ==========
    print("Stage 02: Fine Tuning")
    unfreeze_model(model = model )
    fine_history = model.fit(
        train_data,
        epochs=args.initial_epochs + args.fine_tune_epochs,
        initial_epoch=feature_history.epoch[-1],
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        callbacks=[tb_callback, ckpt_callback, early_stop, best_chkpt_callback]
    )

    # ========== 6. Save Model ==========
    save_model(model, filepath=args.model_save_path)
    print("Model saved!")

    # ========== 7. Plot Curves ==========
    combine_history(feature_history, fine_history)

    if args.save_training_curves:
        print(f"Save loss curves into {args.save_training_curves} path")
        plot_combined_history(feature_history , fine_history, savefig=True, save_path=args.save_training_curves)
        print("loss curves saved!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Vision 101 Training")
    parser.add_argument("--train_dir", type=str,default = "data/train" , help="Path to training data")
    parser.add_argument("--test_dir", type=str, default = "data/test", help="Path to test data")
    parser.add_argument("--initial_epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=32,help="adjust the number with with your VRAM size ") 
    parser.add_argument("--log_dir", type=str, default="Tensorflow_hub_log")
    parser.add_argument("--experiment_name", type=str, default="EfficientnetV2B0")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--model_save_path", type=str, default="models/food_classifier.keras")
    parser.add_argument("--confusion_matrix", action="store_true", default=True)
    parser.add_argument("--savefig" , action="store_true",help="save training image s to images/folders")
    parser.add_argument("--save_training_curves", type=str, default="images/combined_training_curves.png",
                        help="save training loss to images/loss_curves.png")

    args = parser.parse_args()
    train(args)