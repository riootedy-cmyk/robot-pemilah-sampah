import os
import shutil
import random
from pathlib import Path

# ==========================================
# KONFIGURASI
# ==========================================

DOWNLOADS = Path(r"C:\Users\LENOVO\Downloads")

# Cari folder dataset yang namanya diawali dengan ini
DATASET_PREFIX = "Dataset waste image classification Organik and Ino"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# Folder output di project VS Code
OUTPUT_DIR = Path("dataset")

# ==========================================
# CARI FOLDER DATASET
# ==========================================

dataset_folders = [
    folder for folder in DOWNLOADS.iterdir()
    if folder.is_dir() and folder.name.startswith(DATASET_PREFIX)
]

if not dataset_folders:
    print("ERROR: Folder dataset tidak ditemukan.")
    print("Cek nama folder di Downloads.")
    exit()

SOURCE_DIR = dataset_folders[0]

print("Dataset ditemukan:")
print(SOURCE_DIR)

# ==========================================
# CARI FOLDER ORGANIC DAN INORGANIC
# ==========================================

organic_candidates = list(SOURCE_DIR.rglob("Organic"))
inorganic_candidates = list(SOURCE_DIR.rglob("Inorganic"))

if not organic_candidates:
    print("ERROR: Folder Organic tidak ditemukan.")
    exit()

if not inorganic_candidates:
    print("ERROR: Folder Inorganic tidak ditemukan.")
    exit()

ORGANIC_DIR = organic_candidates[0]
INORGANIC_DIR = inorganic_candidates[0]

print("\nFolder Organic ditemukan:")
print(ORGANIC_DIR)

print("\nFolder Inorganic ditemukan:")
print(INORGANIC_DIR)

if not ORGANIC_DIR.exists():
    print("ERROR: Folder Organic tidak ditemukan.")
    exit()

if not INORGANIC_DIR.exists():
    print("ERROR: Folder Inorganic tidak ditemukan.")
    exit()

# ==========================================
# BUAT FOLDER OUTPUT
# ==========================================

folders = [
    OUTPUT_DIR / "train" / "organik",
    OUTPUT_DIR / "train" / "non_organik",
    OUTPUT_DIR / "validation" / "organik",
    OUTPUT_DIR / "validation" / "non_organik"
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# FUNGSI MEMBAGI DATASET
# ==========================================

def split_dataset(source_folder, train_folder, validation_folder):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    images = []

    for extension in extensions:
        images.extend(source_folder.glob(extension))

    # Hilangkan kemungkinan duplikat
    images = list(set(images))

    random.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)

    train_images = images[:train_count]
    validation_images = images[train_count:]

    print("\nFolder:", source_folder.name)
    print("Total:", total)
    print("Training:", len(train_images))
    print("Validation:", len(validation_images))

    # Copy training
    for image in train_images:
        destination = train_folder / image.name
        shutil.copy2(image, destination)

    # Copy validation
    for image in validation_images:
        destination = validation_folder / image.name
        shutil.copy2(image, destination)

    return len(train_images), len(validation_images)


# ==========================================
# RANDOM SEED
# ==========================================

random.seed(RANDOM_SEED)

# ==========================================
# PROSES ORGANIK
# ==========================================

organic_train, organic_val = split_dataset(
    ORGANIC_DIR,
    OUTPUT_DIR / "train" / "organik",
    OUTPUT_DIR / "validation" / "organik"
)

# ==========================================
# PROSES NON-ORGANIK
# ==========================================

inorganic_train, inorganic_val = split_dataset(
    INORGANIC_DIR,
    OUTPUT_DIR / "train" / "non_organik",
    OUTPUT_DIR / "validation" / "non_organik"
)

# ==========================================
# HASIL AKHIR
# ==========================================

print("\n====================================")
print("DATASET BERHASIL DISIAPKAN!")
print("====================================")

print("\nORGANIK")
print("Train      :", organic_train)
print("Validation :", organic_val)

print("\nNON-ORGANIK")
print("Train      :", inorganic_train)
print("Validation :", inorganic_val)

print("\nLokasi dataset:")
print(OUTPUT_DIR.absolute())

print("\nStruktur:")
print("dataset/")
print("├── train/")
print("│   ├── organik/")
print("│   └── non_organik/")
print("│")
print("└── validation/")
print("    ├── organik/")
print("    └── non_organik/")