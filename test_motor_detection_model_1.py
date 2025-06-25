import torch
import cv2
import os

# Load model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

# Folder gambar uji
img_folder = 'test_images'
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)

# Loop gambar
for img_name in os.listdir(img_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)

    # Konversi ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi objek
    results = model(img_rgb)

    # Dapatkan hasil
    results.print()  # Tampilkan di terminal
    results.save(save_dir=output_folder)  # Simpan ke folder
