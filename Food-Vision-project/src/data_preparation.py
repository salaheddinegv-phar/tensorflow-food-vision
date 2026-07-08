
import tensorflow as f
from tensorflow.keras.preprocessing import image_dataset_from_directory
def create_data_loader(train_dir ,test_dir , image_size = (224,224) , batch_size = 32):
    train_data =  image_dataset_from_directory(train_dir ,
                                              label_mode = "categorical" ,
                                              image_size = image_size ,
                                              batch_size = batch_size)
    test_data =  image_dataset_from_directory(test_dir ,
                                              label_mode = "categorical",
                                              image_size = image_size,
                                              batch_size = batch_size)
  