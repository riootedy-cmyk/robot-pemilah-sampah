import cv2
import tensorflow as tf
import numpy as np

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "models/waste_classifier.keras"
)

# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    exit()

print("Kamera aktif.")
print("Tekan Q untuk keluar.")

# =========================
# REAL-TIME DETECTION
# =========================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Gagal membaca kamera.")
        break

    # Resize sesuai input MobileNetV2
    image = cv2.resize(
        frame,
        (224, 224)
    )

    image = image.astype("float32")

    image = np.expand_dims(
        image,
        axis=0
    )

    # Jangan preprocess_input di sini.
    # Preprocessing sudah ada di dalam model.

    prediction = model.predict(
        image,
        verbose=0
    )[0][0]

    # =========================
    # CLASSIFICATION
    # =========================

    if prediction >= 0.5:
        label = "ORGANIK"
        confidence = prediction
    else:
        label = "NON-ORGANIK"
        confidence = 1 - prediction

    # =========================
    # DISPLAY
    # =========================

    text = f"{label} - {confidence * 100:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Robot Pemilah Sampah - AI",
        frame
    )

    # Q = keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()