import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def augment(image, label):
    """
    Robust augmentation for a 10% Food101 subset.
    Handles 4D batched tensors cleanly.
    """
    # 1. Flips (TensorFlow handles batched 4D tensors automatically here)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # 2. Random Crop & Resize (Adjusted for 4D Batch Dimension)
    original_shape = tf.shape(image) # This will be [batch_size, height, width, channels]
    batch_size = original_shape[0]
    
    scale = tf.random.uniform([], 0.85, 1.0)
    new_h = tf.cast(tf.cast(original_shape[1], tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(original_shape[2], tf.float32) * scale, tf.int32)
    
    # Notice we add batch_size at the front, matching the 4D shape!
    image = tf.image.random_crop(image, size=[batch_size, new_h, new_w, 3])
    
    # Resize back to the standard 224x224 shape
    image = tf.image.resize(image, [original_shape[1], original_shape[2]])

    # 3. Brightness, Contrast & Saturation
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    
    # Clip back to standard float pixel boundaries
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
        batch_size=batch_size
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
    # Order: normalize FIRST -> then augment on [-1, 1] -> cache -> prefetch
    train_data = train_data.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    train_data = train_data.map(augment, num_parallel_calls=AUTOTUNE)
    train_data = train_data.map(normalize, num_parallel_calls=AUTOTUNE) # Now scale down to [-1, 1]
    train_data = train_data.cache().prefetch(AUTOTUNE)
    
    # ========== TEST PIPELINE ==========
    # No augmentation, just normalize
    test_data = test_data.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.cache().prefetch(AUTOTUNE)
    
    return train_data, test_data, num_classes , class_names