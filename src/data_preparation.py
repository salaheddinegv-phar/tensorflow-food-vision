import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def augment(image, label):
    """
    Robust augmentation for batched 4D tensors.
    Works with [batch_size, height, width, channels].
    """
    # 1. Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # 2. Random Zoom using resize (batch-safe)
    original_shape = tf.shape(image)
    
    scale = tf.random.uniform([], 0.85, 1.0)
    new_h = tf.cast(tf.cast(original_shape[1], tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(original_shape[2], tf.float32) * scale, tf.int32)
    
    image = tf.image.resize(image, [new_h, new_w])
    image = tf.image.resize_with_crop_or_pad(image, original_shape[1], original_shape[2])

    # 3. Brightness, Contrast & Saturation
    image = tf.image.random_brightness(image, max_delta=16.0 / 255.0)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    
    return image, label

def normalize(image, label):
    """
    EfficientNetV2 preprocessing: [0, 255] -> [-1, 1]
    """
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image, label

def create_data_loader(train_dir, test_dir, image_size=(224, 224), batch_size=32):
    # Load raw [0, 255] data
    train_data = image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        image_size=image_size,
        shuffle=True,
        batch_size=batch_size,
        interpolation="bilinear"
    )
    
    test_data = image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        image_size=image_size,
        shuffle=False,
        batch_size=batch_size
    )
    
    class_names = train_data.class_names
    num_classes = len(class_names)
    AUTOTUNE = tf.data.AUTOTUNE
    
    # ========== TRAIN PIPELINE ==========
    # FIXED: Added buffer_size to prefetch for better throughput
    train_data = train_data.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    train_data = train_data.map(augment, num_parallel_calls=AUTOTUNE)
    train_data = train_data.map(normalize, num_parallel_calls=AUTOTUNE)
    # FIXED: Added buffer_size=AUTOTUNE for better prefetching
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    # ========== TEST PIPELINE ==========
    test_data = test_data.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=AUTOTUNE)

    # FIXED: Return class_names as 4th value
    return train_data, test_data, num_classes, class_names