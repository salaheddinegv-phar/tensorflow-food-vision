import matplotlib.pyplot as plt
import argparse
import pickle

def plot_combined_history(feature_history, fine_history, savefig=False, save_path="images/combined_training_curves.png"):
    """
    Plots feature extraction and fine-tuning histories as one continuous curve.
    feature_history and fine_history are DICTIONARIES (from pickle), not History objects.
    """
    # Combine epochs
    feat_epochs = list(range(1, len(feature_history['loss']) + 1))
    fine_start = len(feat_epochs)
    fine_epochs = list(range(fine_start + 1, fine_start + len(fine_history['loss']) + 1))
    
    # Combine losses - these are now direct lists, not .history['loss']
    all_loss = feature_history['loss'] + fine_history['loss']
    all_val_loss = feature_history.get('val_loss', []) + fine_history.get('val_loss', [])
    all_acc = feature_history['accuracy'] + fine_history['accuracy']
    all_val_acc = feature_history.get('val_accuracy', []) + fine_history.get('val_accuracy', [])
    
    all_epochs = feat_epochs + fine_epochs
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History: Feature Extraction → Fine Tuning', fontsize=16, fontweight='bold')
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(all_epochs, all_loss, 'o-', color='#E74C3C', linewidth=2, label='Training Loss', markersize=4)
    if all_val_loss:
        ax1.plot(all_epochs, all_val_loss, 's-', color='#3498DB', linewidth=2, label='Validation Loss', markersize=4)
    ax1.axvline(x=fine_start + 0.5, color='gray', linestyle='--', alpha=0.7, label='Unfreeze')
    ax1.set_title('Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Accuracy plot
    ax2 = axes[1]
    ax2.plot(all_epochs, all_acc, 'o-', color='#2ECC71', linewidth=2, label='Training Accuracy', markersize=4)
    if all_val_acc:
        ax2.plot(all_epochs, all_val_acc, 's-', color='#9B59B6', linewidth=2, label='Validation Accuracy', markersize=4)
    ax2.axvline(x=fine_start + 0.5, color='gray', linestyle='--', alpha=0.7, label='Unfreeze')
    ax2.set_title('Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    if savefig:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_history", type=str, required=True, help="Path to feature_history.pkl")
    parser.add_argument("--fine_history", type=str, required=True, help="Path to fine_history.pkl")
    parser.add_argument("--save_path", type=str, default="images/combined_training_curves.png")
    args = parser.parse_args()
    
    # Load the pickled dictionaries
    with open(args.feature_history, "rb") as f:
        feature_hist = pickle.load(f)
    
    with open(args.fine_history, "rb") as f:
        fine_hist = pickle.load(f)
    
    # Pass the dictionaries (not History objects)
    plot_combined_history(feature_hist, fine_hist, savefig=True, save_path=args.save_path)
