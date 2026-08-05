
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


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(100, 100), 
                         text_size=20, norm=False, savefig=False, save_path=None):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        classes: List of class names (if None, integers are used)
        figsize: Figure size
        text_size: Font size for labels
        norm: Whether to show normalized percentages
        savefig: Whether to save the figure
        save_path: Custom path to save the figure (default: images/confusion_matrix.png)
    """
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    # Normalized version (row-wise)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use imshow instead of matshow for better control
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    fig.colorbar(im, ax=ax)
    
    # Set labels
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(n_classes)
    
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right", fontsize=text_size)
    plt.setp(ax.get_yticklabels(), fontsize=text_size)
    
    # Threshold for text color
    threshold = (cm.max() + cm.min()) / 2
    
    # Add text annotations
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        if norm:
            text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
        else:
            text = f"{cm[i, j]}"
        
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(j, i, text, 
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=text_size-4)
    
    plt.tight_layout()
    
    # Save figure
    if savefig:
        if save_path is None:
            save_path = "images/confusion_matrix.png"   # Relative to project root
            
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to → {save_path}")
    
    plt.show()
    return fig
  
def plot_combined_history(*histories, savefig=False, save_path="images/combined_training_curves_full.png"):
    """
    Plots multiple training histories (feature extraction + multiple fine-tuning stages)
    as one continuous curve with unfreeze markers.
    
    Usage:
        plot_combined_history(feature_history, fine_history_1, fine_history_2, fine_history_3)
    """
    # Combine all histories
    all_loss = []
    all_val_loss = []
    all_acc = []
    all_val_acc = []
    all_epochs = []
    
    unfreeze_points = []  # Points where unfreezing happened
    current_epoch = 0
    
    for i, history in enumerate(histories):
        if history is None:
            continue
            
        num_epochs = len(history.history['loss'])
        epochs = list(range(current_epoch + 1, current_epoch + num_epochs + 1))
        
        all_loss.extend(history.history['loss'])
        all_val_loss.extend(history.history.get('val_loss', []))
        all_acc.extend(history.history['accuracy'])
        all_val_acc.extend(history.history.get('val_accuracy', []))
        all_epochs.extend(epochs)
        
        # Mark unfreeze point (except for first history = feature extraction)
        if i > 0:
            unfreeze_points.append(current_epoch + 0.5)
        
        current_epoch += num_epochs
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History: Feature Extraction → Fine Tuning', fontsize=16, fontweight='bold')
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(all_epochs, all_loss, 'o-', color='#E74C3C', linewidth=2, label='Training Loss', markersize=3)
    if all_val_loss:
        ax1.plot(all_epochs, all_val_loss, 's-', color='#3498DB', linewidth=2, label='Validation Loss', markersize=3)
    
    # Add unfreeze markers
    for point in unfreeze_points:
        ax1.axvline(x=point, color='gray', linestyle='--', alpha=0.7)
    
    # Add legend for unfreeze stages
    if len(unfreeze_points) >= 1:
        ax1.axvline(x=unfreeze_points[0], color='gray', linestyle='--', alpha=0.7, label='Unfreeze 10 layers')
    if len(unfreeze_points) >= 2:
        ax1.axvline(x=unfreeze_points[1], color='orange', linestyle='--', alpha=0.7, label='Unfreeze 30 layers')
    if len(unfreeze_points) >= 3:
        ax1.axvline(x=unfreeze_points[2], color='green', linestyle='--', alpha=0.7, label='Unfreeze ALL')
    
    ax1.set_title('Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Accuracy plot
    ax2 = axes[1]
    ax2.plot(all_epochs, all_acc, 'o-', color='#2ECC71', linewidth=2, label='Training Accuracy', markersize=3)
    if all_val_acc:
        ax2.plot(all_epochs, all_val_acc, 's-', color='#9B59B6', linewidth=2, label='Validation Accuracy', markersize=3)
    
    # Add same unfreeze markers
    for point in unfreeze_points:
        ax2.axvline(x=point, color='gray', linestyle='--', alpha=0.7)
    
    if len(unfreeze_points) >= 1:
        ax2.axvline(x=unfreeze_points[0], color='gray', linestyle='--', alpha=0.7, label='Unfreeze 10 layers')
    if len(unfreeze_points) >= 2:
        ax2.axvline(x=unfreeze_points[1], color='orange', linestyle='--', alpha=0.7, label='Unfreeze 30 layers')
    if len(unfreeze_points) >= 3:
        ax2.axvline(x=unfreeze_points[2], color='green', linestyle='--', alpha=0.7, label='Unfreeze ALL')
    
    ax2.set_title('Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    if savefig:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
    return fig


def plot_top_confusions(y_true, y_pred, classes, top_n=20, figsize=(14, 10), 
                        savefig=False, save_path="images/top_confusions_full.png"):
    """
    Plots only the most confused class pairs (off-diagonal elements).
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Get off-diagonal confusion scores (where model gets it wrong)
    # Create a mask for off-diagonal elements
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)  # Remove correct predictions
    
    # Find top confused pairs
    # Get indices sorted by confusion score (descending)
    flat_indices = np.argsort(off_diag.flatten())[::-1]
    top_pairs = []
    
    for idx in flat_indices:
        if len(top_pairs) >= top_n:
            break
        i, j = np.unravel_index(idx, cm.shape)
        if i != j and off_diag[i, j] > 0:  # Only wrong predictions
            top_pairs.append((i, j, off_diag[i, j]))
    
    # Extract class names and scores
    true_classes = [classes[i] for i, j, _ in top_pairs]
    pred_classes = [classes[j] for i, j, _ in top_pairs]
    scores = [score for _, _, score in top_pairs]
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_pairs))
    bars = ax.barh(y_pos, scores, color='#E74C3C', alpha=0.7)
    
    # Add labels: "true_class → pred_class"
    labels = [f"{true} → {pred}" for true, pred in zip(true_classes, pred_classes)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Highest confusion at top
    
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title(f'Top {top_n} Most Confused Class Pairs', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{int(score)}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if savefig:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
    return fig