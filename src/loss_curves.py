{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "843cc308",
   "metadata": {},
   "outputs": [],
   "source": [
    "import  matplotlib.pyplot as plt\n",
    "import os "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d5b3391",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_combined_history(feature_history, fine_history, savefig=False, save_path=\"images/combined_training_curves.png\"):\n",
    "    \"\"\"\n",
    "    Plots feature extraction and fine-tuning histories as one continuous curve.\n",
    "    \"\"\"\n",
    "    # Combine epochs\n",
    "    feat_epochs = list(range(1, len(feature_history.history['loss']) + 1))\n",
    "    fine_start = len(feat_epochs)\n",
    "    fine_epochs = list(range(fine_start + 1, fine_start + len(fine_history.history['loss']) + 1))\n",
    "    \n",
    "    # Combine losses\n",
    "    all_loss = feature_history.history['loss'] + fine_history.history['loss']\n",
    "    all_val_loss = feature_history.history.get('val_loss', []) + fine_history.history.get('val_loss', [])\n",
    "    all_acc = feature_history.history['accuracy'] + fine_history.history['accuracy']\n",
    "    all_val_acc = feature_history.history.get('val_accuracy', []) + fine_history.history.get('val_accuracy', [])\n",
    "    \n",
    "    all_epochs = feat_epochs + fine_epochs\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "    fig.suptitle('Training History: Feature Extraction → Fine Tuning', fontsize=16, fontweight='bold')\n",
    "    \n",
    "    # Loss plot\n",
    "    ax1 = axes[0]\n",
    "    ax1.plot(all_epochs, all_loss, 'o-', color='#E74C3C', linewidth=2, label='Training Loss', markersize=4)\n",
    "    if all_val_loss:\n",
    "        ax1.plot(all_epochs, all_val_loss, 's-', color='#3498DB', linewidth=2, label='Validation Loss', markersize=4)\n",
    "    ax1.axvline(x=fine_start + 0.5, color='gray', linestyle='--', alpha=0.7, label='Unfreeze')\n",
    "    ax1.set_title('Loss', fontsize=14)\n",
    "    ax1.set_xlabel('Epochs')\n",
    "    ax1.set_ylabel('Loss')\n",
    "    ax1.legend()\n",
    "    ax1.grid(True, linestyle='--', alpha=0.4)\n",
    "    \n",
    "    # Accuracy plot\n",
    "    ax2 = axes[1]\n",
    "    ax2.plot(all_epochs, all_acc, 'o-', color='#2ECC71', linewidth=2, label='Training Accuracy', markersize=4)\n",
    "    if all_val_acc:\n",
    "        ax2.plot(all_epochs, all_val_acc, 's-', color='#9B59B6', linewidth=2, label='Validation Accuracy', markersize=4)\n",
    "    ax2.axvline(x=fine_start + 0.5, color='gray', linestyle='--', alpha=0.7, label='Unfreeze')\n",
    "    ax2.set_title('Accuracy', fontsize=14)\n",
    "    ax2.set_xlabel('Epochs')\n",
    "    ax2.set_ylabel('Accuracy')\n",
    "    ax2.set_ylim(0, 1.05)\n",
    "    ax2.legend()\n",
    "    ax2.grid(True, linestyle='--', alpha=0.4)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    \n",
    "    if savefig:\n",
    "        import os\n",
    "        os.makedirs(os.path.dirname(save_path), exist_ok=True)\n",
    "        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')\n",
    "        print(f\"✅ Saved to: {save_path}\")\n",
    "    \n",
    "    plt.show()\n",
    "    return fig\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
