
import sys 
import os

from matplotlib.pyplot import savefig 
  
sys.path.append(os.path.abspath("/content/tensorflow-food-vision/Food-Vision-project/src"))

from src.data_preparation import create_data_loader
from src.model_utils import create_model , save_model , unfreeze_model
from src.evaluation import plot_training_curves , make_confusion_matrix
from helper_functions import create_tensorboard_callback , create_checkpoint_callback , combine_history

train_dir = "/content/tensorflow-food-vision/Food-Vision-project/data/101_food_classes_10_percent/train"  ## write your own train diretory
test_dir = "/content/tensorflow-food-vision/Food-Vision-project/data/101_food_classes_10_percent/test"  ## write your own test diretory
print(f" Train_data directory : {train_dir}")
print(f" test_data directory : {test_dir}")

train_data, test_data = create_data_loader(train_dir , test_dir)
print(f"Data loaded!! ")

efficientnet_model = create_model(num_classes = train_data.num_classes)
print(f"EfficientnetV2B0 model created succesfully!")

efficientnet_model.summary()
print(f"Model summary printed successfully!")

  ## create callbacks :
create_tensorboard_callback(dir_name = "Tensorflow_hub_log",
                              experiment_name = "EfficientnetV2BB0_10%Data")
create_checkpoint_callback(checkpoint_path = "/content/tensorflow-food-vision/Food-Vision-project/models/checkpoints" )
print(f"Callbacks created successfully!")

initial_epochs = 15
## Training our model
## ======Stage 01==========
print(f"Training Stage 01 : =====Feature_extraction======")
feature_extraction_history = efficientnet_model.fit(train_data,
                                                     epochs = initial_epochs,
                                                     steps_per_epoch = len(train_data),
                                                     validation_data = test_data,
                                                     validation_steps = int(0.25 * len(test_data)),
                                                     callbacks = [create_tensorboard_callback(dir_name = "Tensorflow_hub_log",
                                                                                             experiment_name = "EfficientnetV2BB0_10%Data"),
                                                                  create_checkpoint_callback(checkpoint_path = "/content/tensorflow-food-vision/Food-Vision-project/models/checkpoints")])


## ======Stage 02==========
print(f"Training Stage 02 : =====Fine_tuning======")
unfreeze_model(efficientnet_model)
fine_tuning_history = efficientnet_model.fit(train_data,
                                              epochs = initial_epochs + 5 ,
                                              initial_epochs = feature_extraction_history.epoch[-1],
                                              steps_per_epoch = len(train_data),
                                              validation_data = test_data,
                                              validation_steps = len(test_data))

## Saving our model
save_model(efficientnet_model)
print(f" Your model saved succesfully !!")

## showing the Changes that happening to train and validation  loss , accuracy when we training our model with feature extraction and fine tuning stages
combine_history(feature_extraction_history , fine_tuning_history)

## plotting Confusion_matrix ==Note==: "before plotting it we it to extract y_labels and y_pred=="pred_classes"
pred_probs = efficientnet_model.predict(test_data , verbose = 1) #setting verbose = 1 to see how much the model take to make prediction
pred_classes = pred_probs.argmax(axis=1)

## making y_labels list :
y_labels = []
for images , labels in test_data.unbatch() :
  y_labels.append(labels.numpy().argmax())

## The extraction of class_names :
class_names = test_data.class_names
print(class_names[:15]) ## seeing the 15 first classes 

## Using make_confusion_matrix function :
make_confusion_matrix(y_labels=y_labels , y_pred=pred_classes , classes=class_names ,figsize = (100,100), text_size=20 ,norm=True, savefig=True)











  