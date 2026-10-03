# ♻️ Robot Pemilah Sampah Organik dan Non-Organik

## Deskripsi

Proyek ini merupakan sistem klasifikasi sampah berbasis Computer Vision untuk membantu robot memilah sampah menjadi dua kategori, yaitu **organik** dan **non-organik**.

Model dibangun menggunakan metode **Transfer Learning** dengan arsitektur **MobileNetV2** yang telah dilatih menggunakan dataset ImageNet.

Hasil klasifikasi dari kamera dapat digunakan sebagai dasar pengendalian aktuator seperti servo pada robot pemilah sampah.

## Tujuan

1. Mengembangkan model Computer Vision untuk mengenali sampah organik dan non-organik.
2. Menerapkan metode Transfer Learning menggunakan MobileNetV2.
3. Menguji kemampuan model dalam mengklasifikasikan gambar sampah.
4. Mengimplementasikan model pada kamera secara real-time sebagai bagian dari konsep robot pemilah sampah.

## Dataset

Dataset terdiri dari:

| Kategori    |  Jumlah |
| ----------- | ------: |
| Organik     |     428 |
| Non-organik |     249 |
| **Total**   | **677** |

Dataset dibagi dengan rasio:

* 80% untuk training
* 20% untuk validation

Sumber dataset:

Organik dan Inorganik Waste Image Classification Dataset — Mendeley Data.

Dataset tidak disertakan seluruhnya di repository karena ukuran file yang besar. Silakan mengunduh dataset dari sumber aslinya dan menyesuaikan struktur folder seperti pada project ini.

## Struktur Dataset

```text
dataset/
├── train/
│   ├── organik/
│   └── non_organik/
│
└── validation/
    ├── organik/
    └── non_organik/
```

## Metode Transfer Learning

Arsitektur yang digunakan:

**MobileNetV2**

Model MobileNetV2 pretrained menggunakan ImageNet digunakan sebagai feature extractor. Layer dasar model dibekukan, kemudian ditambahkan Global Average Pooling, Dropout, dan Dense layer untuk klasifikasi dua kelas.

Alur sistem:

```text
Gambar Sampah
      ↓
Preprocessing
      ↓
MobileNetV2
      ↓
Feature Extraction
      ↓
Classification Layer
      ↓
┌───────────────┐
│               │
Organik    Non-Organik
│               │
└───────┬───────┘
        ↓
   Robot Pemilah
```

## Hasil Training

Training dilakukan selama 10 epoch.

Hasil akhir:

| Parameter           |  Hasil |
| ------------------- | -----: |
| Training Accuracy   | 96.67% |
| Validation Accuracy | 96.32% |
| Training Loss       | 0.1300 |
| Validation Loss     | 0.1324 |

Grafik training dan validation accuracy tersedia pada:

```text
accuracy.png
```

## Pengujian

Pengujian tambahan menggunakan data validation menghasilkan:

```text
Total gambar : 272
Benar        : 172
Salah        : 100
Accuracy     : 63.24%
```

Hasil pengujian tambahan tersebut dicantumkan sebagai evaluasi terpisah dari validation accuracy selama proses training.

## File Project

```text
robot-pemilah-sampah/
│
├── dataset/
│   ├── train/
│   └── validation/
│
├── models/
│   └── waste_classifier.keras
│
├── train.py
├── prepare_dataset.py
├── predict.py
├── test_model.py
├── camera.py
├── accuracy.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalasi

Install library yang diperlukan:

```bash
pip install -r requirements.txt
```

## Training Model

Untuk melakukan training:

```bash
python train.py
```

Model akan disimpan sebagai:

```text
models/waste_classifier.keras
```

## Prediksi Gambar

Untuk melakukan klasifikasi satu gambar:

```bash
python predict.py nama_gambar.jpg
```

Contoh:

```bash
python predict.py botol.jpg
```

## Pengujian Model

Untuk menguji model menggunakan dataset validation:

```bash
python test_model.py
```

## Kamera Real-Time

Untuk menjalankan klasifikasi menggunakan webcam:

```bash
python camera.py
```

Tekan tombol **Q** untuk keluar dari aplikasi kamera.

## Konsep Implementasi Robot

Pada implementasi robot secara fisik, kamera digunakan untuk mengambil gambar sampah. Model AI kemudian melakukan klasifikasi.

```text
Camera
   ↓
MobileNetV2
   ↓
Klasifikasi
   ↓
Organik / Non-Organik
   ↓
Arduino / ESP32
   ↓
Servo Motor
   ↓
Tempat Sampah
```

Servo dapat diarahkan ke posisi yang berbeda berdasarkan hasil klasifikasi.

## Teknologi

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Matplotlib
* Computer Vision
* Transfer Learning

## Pengembangan Selanjutnya

Pengembangan berikutnya dapat meliputi:

* Penambahan jumlah dan variasi dataset.
* Pengujian menggunakan objek sampah secara real-time.
* Peningkatan kemampuan generalisasi model.
* Integrasi model dengan Arduino atau ESP32.
* Pengendalian servo untuk proses pemilahan otomatis.
* Pengembangan sistem robot pemilah sampah secara keseluruhan.

## Author

**Rio Tedy Adhyaksa Situmorang**

Project — Klasifikasi Sampah Organik dan Non-Organik
