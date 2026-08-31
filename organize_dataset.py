import os
import shutil

SOURCE = "dataset/val"
OUTPUT = "dataset/val_disease"

wanted = {
    "Corn_(maize)___Common_rust_": "Corn___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn___Northern_Leaf_Blight",
    "Corn_(maize)___healthy": "Corn___healthy",

    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",

    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___healthy": "Tomato___healthy"
}

os.makedirs(OUTPUT, exist_ok=True)

for source_name, destination_name in wanted.items():

    destination = os.path.join(OUTPUT, destination_name)
    os.makedirs(destination, exist_ok=True)

    found = False

    for crop in ["corn", "potato", "tomato"]:

        crop_path = os.path.join(SOURCE, crop)

        if not os.path.exists(crop_path):
            continue

        for root, dirs, files in os.walk(crop_path):

            if os.path.basename(root) == source_name:

                print(f"\nFound: {source_name}")
                print(f"Copying {len(files)} images...")

                for file in files:

                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(
                        destination,
                        file
                    )

                    shutil.copy2(
                        source_file,
                        destination_file
                    )

                print(f"Done -> {destination_name}")
                found = True
                break

        if found:
            break

    if not found:
        print(f"\nNOT FOUND: {source_name}")


print("\n==============================")
print("Dataset organization complete!")
print("==============================")
