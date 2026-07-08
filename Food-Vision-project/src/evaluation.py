
import itertools 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def plot_training_curves(history):
    """Plot training curves (accuracy and loss)"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def make_confusion_metrix(y_true , y_pred , classes = None , figsize = (10,10), text_size = 20 , norm = False ,savefig = False):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).
  
    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
      """
    
    cm = confusion_matrix(y_true , y_pred)
    cm_norm = cm.astype("float")/cm.sum(axis = 1)[:,np.newaxis] ## explain what's on over here ::
                                                                ## i wrote this cm_norm i order the user wanna see the performance of the model on predicting classes with precentage 
                                                                ## not with row counts 

    n_classes = cm.shape[0]
    fig , ax  = plt.subplot(figsize = figsize)
    cax = plt.matshow(cm , cmap = plt.cm.Greens) ## Display metrix with the corsponding color 
    fig.colorbar(cax)

    if classes:
      labels = classes
    else :
      labels = np.arange(axis = 0)

    ax.set(title = "Confusion metrix",
           xlabel = "true label",
           ylabel = "Predicted label",
           xticks = np.arange(n_classes),
           yticks = np.arange(n_classes),
           xticks_labels = labels ,
           yticks_labels = labels
           )
    ## making rotation for classes situated on x-axis to be clear for reading it 
    plt.xticks(rotaion = 70 , fontsize = text_size)
    plt.yticks(fontsize = text_size)

    threshold = (cm.max()+cm.min())/2 

    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
      if norm :
        plt.text(j , i , f"{cm[i , j]} ({cm_norm[i , j]*100:.1f}%)",
                 horizontalalignement = "center",
                 color = "white" if cm[i , j] > threshold else "black")
      else : 
        plt.text(j , i , f"{cm[i , j]}" ,
                 horizontalalignement = "center",
                 color = "white" if cm[i , j] > threshold else "black")
    ## saving figure to imgages folder hint this will work after setting savefig = True
    save_path = "/content/tensorflow-food-vision/Food-Vision-project/images/confusion_metrix.png"
    if savefig : 
      fig.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"✅ Confusion matrix saved to → {save_path}")
  