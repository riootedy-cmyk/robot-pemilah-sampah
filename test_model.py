import tensorflow as tf
import numpy as np
from pathlib import Path

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/waste_classifier.keras"
)

# ==========================================
# FUNGSI PREDIKSI
# ==========================================

def predict_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    if prediction >= 0.5:
        predicted_class = "organik"
        confidence = prediction
    else:
        predicted_class = "non_organik"
        confidence = 1 - prediction

    return predicted_class, confidence


# ==========================================
# DATA VALIDATION
# ==========================================

folders = {
    "non_organik": Path("dataset/validation/non_organik"),
    "organik": Path("dataset/validation/organik")
}

total = 0
correct = 0

hasil = {
    "non_organik": {
        "benar": 0,
        "salah": 0
    },
    "organik": {
        "benar": 0,
        "salah": 0
    }
}

# ==========================================
# TEST SEMUA GAMBAR
# ==========================================

for true_class, folder in folders.items():

    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        images.extend(folder.glob(ext))

    print()
    print("=" * 60)
    print("KELAS SEBENARNYA:", true_class.upper())
    print("Jumlah gambar:", len(images))
    print("=" * 60)

    for image_path in images:

        predicted_class, confidence = predict_image(
            image_path
        )

        total += 1

        if predicted_class == true_class:

            correct += 1

            hasil[true_class]["benar"] += 1

            status = "BENAR"

        else:

            hasil[true_class]["salah"] += 1

            status = "SALAH"

        print(
            f"{image_path.name} -> "
            f"{predicted_class.upper()} "
            f"{confidence * 100:.2f}% "
            f"[{status}]"
        )


# ==========================================
# HASIL PER KELAS
# ==========================================

print()
print("=" * 60)
print("HASIL PER KELAS")
print("=" * 60)

for class_name in ["non_organik", "organik"]:

    benar = hasil[class_name]["benar"]
    salah = hasil[class_name]["salah"]

    jumlah = benar + salah

    accuracy = (
        benar / jumlah * 100
        if jumlah > 0 else 0
    )

    print()
    print(class_name.upper())
    print("Benar     :", benar)
    print("Salah     :", salah)
    print("Accuracy  :", f"{accuracy:.2f}%")


# ==========================================
# HASIL AKHIR
# ==========================================

overall_accuracy = (
    correct / total * 100
    if total > 0 else 0
)

print()
print("=" * 60)
print("HASIL PENGUJIAN SELURUH VALIDATION DATASET")
print("=" * 60)

print("Total gambar :", total)
print("Benar        :", correct)
print("Salah        :", total - correct)
print("Accuracy     :", f"{overall_accuracy:.2f}%")

print("=" * 60)