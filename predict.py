import tensorflow as tf
import numpy as np
import sys
import os

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "models/waste_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# NAMA KELAS
# =========================

# image_dataset_from_directory mengurutkan alfabetis:
# non_organik = 0
# organik = 1

class_names = [
    "non_organik",
    "organik"
]

# =========================
# CEK INPUT
# =========================

if len(sys.argv) < 2:
    print("===================================")
    print("PROGRAM PREDIKSI SAMPAH")
    print("===================================")
    print("Cara penggunaan:")
    print("python predict.py nama_gambar.jpg")
    print()
    print("Contoh:")
    print("python predict.py botol.jpg")
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("ERROR: Gambar tidak ditemukan!")
    print("Path:", image_path)
    sys.exit()

# =========================
# LOAD GAMBAR
# =========================

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# =========================
# PREPROCESS
# =========================

image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# =========================
# PREDIKSI
# =========================

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]

# =========================
# HASIL
# =========================

if prediction >= 0.5:
    class_name = "ORGANIK"
    confidence = prediction
else:
    class_name = "NON-ORGANIK"
    confidence = 1 - prediction

print()
print("===================================")
print("       HASIL DETEKSI SAMPAH")
print("===================================")
print("Gambar     :", image_path)
print("Kelas      :", class_name)
print("Confidence :", f"{confidence * 100:.2f}%")
print("===================================")