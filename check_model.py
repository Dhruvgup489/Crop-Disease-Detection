import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/crop_disease_model.keras"
VAL_DIR = "dataset/val_disease"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

print("Loading validation dataset...")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = val_dataset.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ==========================================
# GET TRUE LABELS
# ==========================================

true_labels = []

for images, labels in val_dataset:
    true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nRunning predictions...")

predictions = model.predict(
    val_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )
)


# ==========================================
# PER CLASS CORRECT / WRONG
# ==========================================

print("\n==========================================")
print("PER CLASS RESULTS")
print("==========================================")

for i, class_name in enumerate(class_names):

    total = np.sum(true_labels == i)

    correct = np.sum(
        (true_labels == i) &
        (predicted_labels == i)
    )

    wrong = total - correct

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"{correct}/{total} correct "
        f"({accuracy:.2f}%)"
    )
