import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/crop_disease_model.keras"
VAL_DIR = "dataset/val_disease"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

OUTPUT_DIR = "static/images"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ==========================================
# LOAD VALIDATION DATASET
# ==========================================

dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = dataset.class_names
display_names = [
    "Corn Rust",
    "Corn Blight",
    "Corn Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]
plt.xticks(
    np.arange(len(class_names)),
    display_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(class_names)),
    display_names
)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)

# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    dataset,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

# ==========================================
# ACTUAL LABELS
# ==========================================

actual_labels = np.concatenate([
    labels.numpy()
    for images, labels in dataset
])

# ==========================================
# CONFUSION MATRIX
# ==========================================

confusion_matrix = tf.math.confusion_matrix(
    actual_labels,
    predicted_labels,
    num_classes=len(class_names)
).numpy()

print("\nCONFUSION MATRIX")
print(confusion_matrix)

# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# DRAW HEATMAP
# ==========================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title(
    "Crop Disease Classification - Confusion Matrix",
    fontsize=16
)

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

# ==========================================
# ADD VALUES
# ==========================================

threshold = confusion_matrix.max() / 2

for i in range(
    confusion_matrix.shape[0]
):

    for j in range(
        confusion_matrix.shape[1]
    ):

        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center"
        )

plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()

# ==========================================
# SAVE IMAGE
# ==========================================

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n==========================================")
print("DONE")
print("==========================================")

print(
    "Confusion matrix saved at:"
)

print(
    OUTPUT_FILE
)
