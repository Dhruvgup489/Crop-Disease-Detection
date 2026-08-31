import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# SETTINGS
# ==============================

TRAIN_DIR = "dataset/train_disease"
VAL_DIR = "dataset/val_disease"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.keras")

# Create model folder if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# DATA AUGMENTATION
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ==============================
# LOAD TRAINING DATA
# ==============================

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# ==============================
# LOAD VALIDATION DATA
# ==============================

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==============================
# SHOW CLASSES
# ==============================

print("\nClasses found:")

for class_name, class_index in train_data.class_indices.items():
    print(class_index, "=", class_name)

# ==============================
# BUILD MODEL
# ==============================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# SHOW MODEL
# ==============================

model.summary()

# ==============================
# TRAIN MODEL
# ==============================

print("\nStarting training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================

model.save(MODEL_PATH)

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")
print(f"Model saved at: {MODEL_PATH}")

print("\nClass mapping:")

for class_name, class_index in train_data.class_indices.items():
    print(f"{class_index}: {class_name}")