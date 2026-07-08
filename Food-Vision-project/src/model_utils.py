
  from tensorflow.keras import layers
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
## creating data augmentation layer to enhance our model performance and prevent overfitting
  data_augementation = Sequential([
      layers.RandomFlip("horizantol_and_vertical"),
      layers.RandomRotation(0.3),
      layers.RandomZoom(0.3),
      layers.RandomHeight(0.3),
      layers.RandomWidth(0.3)],
      name = "data_augementation")
  def create_model(input_shape = (224,224,3) ,num_classes = train_data.num_classes):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False,   ## load pre-trained model(EfficientNetV2B0)
                                                                        weights = "imagenet",
                                                                        input_shape = input_shape)
    base_model.trainable = False ## freeze the base model layers
    ## creating head layer for our base model
    inputs = layers.Input(shape = input_shape , name = "input_layer")
    x = data_augementation(inputs)
    x = base_model(x , training = False)
    #
    x = layers.GlobalAveragePooling2D(name = "Global_average_pooling_layer")(x)
    # Regularization using Droput layer (Role : "prevent overfitting ")
    x = layers.Dropout(0.3)(x)
    # output layer
    output = layers.Dense(num_classes, activation  = "softmax" , name = "ouput_layer")(x)
    model = tf.keras.Model(inputs , output)
    model.compile(loss = "categorical_crossentropy",
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = ["accuracy"])
    return model

    efficientnet_model = create_model()
    efficientnet_model.summary()

  def save_model(model , path = "/content/tensorflow-food-vision/Food-Vision-project/models"):
    model.save(path)
    print(f"Model saved succeesfully at {path}")


  def unfreeze_model(model , num_layers = 10):
    efficientnet_base_model = model.layers[2] ## selecting the base model from our full model (view the summary of the model )
    efficientnet_base_model.trainable = True
    for layer in efficientnet_base_model[:-num_layers]:
      efficientnet_base_model.trainable = False ""unfreeze the last 10 layer in our model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = tf.keras.optimizers.Adam(1e-4), ## change the learning rate with *10 much small than the first time
                                                              ## to prevent our model losing the general feature that has been extracted in feature extraction phase

                              metrics = ["accuracy"])

    print(f"✅ Unfroze last {num_layers} layers. Model ready for fine-tuning.")
    return model

  