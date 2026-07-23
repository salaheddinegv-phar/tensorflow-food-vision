from src.data_preparation import create_data_loader
from src.model_utils import create_model

# Load data
train_data, test_data, num_classes, class_names = create_data_loader(
    "data/train", "data/test", batch_size=32
)

# Check ONE batch — iterate normally, don't use .take(1) on cached data
for images, labels in train_data:
    print("Image shape:", images.shape)
    print("Pixel range:", images.numpy().min(), "to", images.numpy().max())
    print("Label shape:", labels.shape)
    print("Label sum:", labels[0].numpy().sum())
    break  # Only check first batch

# Check model
model = create_model(num_classes=num_classes)
pred = model.predict(test_data, steps=1)  # Use steps=1 instead of .take(1)
print("Prediction shape:", pred.shape)
print("Sample prediction:", pred[0][:5])
print("Sum of probabilities:", pred[0].sum())