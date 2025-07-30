import os
import cv2
import json

# Шляхи
input_dir = "jpg_discovery"
output_dir = "rois"
roi_json_file = "rois.json"

# Завантаження списку ROI
with open(roi_json_file, "r") as f:
    roi_data = json.load(f)
    units = roi_data.get("units", "absolute").lower()  # За замовчуванням absolute
    rois = roi_data.get("rois", [])
if units == "relative":
    print("Realtive units are used, images can be different sizes")
              
# Створення головної папки для збереження ROI
os.makedirs(output_dir, exist_ok=True)

# Прохід по всіх зображеннях
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        name, _ = os.path.splitext(filename)
        img_path = os.path.join(input_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Неможливо прочитати зображення {img_path}, пропущено.")
            continue

        img_height, img_width = img.shape[:2]

        for idx, roi in enumerate(rois):
            roi_type = roi.get("type", "unknown")

            x = roi.get("x", 0)
            y = roi.get("y", 0)
            w = roi.get("w", 0)
            h = roi.get("h", 0)

            # Перевірка одиниць виміру
            if units == "relative":
                x = int(x * img_width)
                y = int(y * img_height)
                w = int(w * img_width)
                h = int(h * img_height)
            else:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

            # Вирізання області
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                print(f"[!] Порожнє ROI у {filename}, тип {roi_type}, пропущено.")
                continue

            # Створити папку для типу
            type_folder = os.path.join(output_dir, roi_type)
            os.makedirs(type_folder, exist_ok=True)

            # Ім’я для ROI-зображення
            roi_filename = f"{name}.jpg"
            roi_path = os.path.join(type_folder, roi_filename)

            # Збереження ROI
            cv2.imwrite(roi_path, crop)

print("✅ Усі ROI успішно вирізано і збережено.")
