
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf

## creating data augmentation layer to enhance ourmodel performance and prevent overfitting

def create_model(input_shape = (224,224,3) ,num_classes = 101):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False,   ## load pre-trained model(EfficientNetV2B0)
                                                                        weights = "imagenet",
                                                                        input_shape = input_shape,
                                                                        include_preprocessing = False)
    base_model.trainable = False ## freeze the base model layers
    ## creating head layer for our base model
    inputs = layers.Input(shape = input_shape , name = "input_layer")
    x = base_model(inputs , training = False)
    x = layers.GlobalAveragePooling2D(name = "Global_average_pooling_layer")(x)
    # Regularization using Droput layer (Role : "prevent overfitting ")
    x = layers.Dropout(0.3)(x)
    # output layer
    output = layers.Dense(num_classes, activation  = "softmax" , name = "output_layer")(x)
    model = tf.keras.Model(inputs , output)
    model.compile(loss = "categorical_crossentropy",
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = ["accuracy"])
    return model


def save_model(model , filepath = "models/food_classifier.keras"):
    model.save(filepath)
    print(f"Model saved succeesfully at {filepath}")


def unfreeze_model(model , num_layers = 10):
    efficientnet_base_model = model.layers[1] ## selecting the base model from our full model (view the summary of the model )
    efficientnet_base_model.trainable = True
    for layer in efficientnet_base_model.layers[:-num_layers]:
      layer.trainable = False #unfreeze the last 10 layer in our model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = tf.keras.optimizers.Adam(1e-5), ## change the learning rate with *10 much small than the first time
                                                              ## to prevent our model losing the general feature that has been extracted in feature extraction phase

                              metrics = ["accuracy"])

    print(f"✅ Unfroze last {num_layers} layers. Model ready for fine-tuning.")
    return model

  