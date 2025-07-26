import os
import cv2
import json
'''
    Цей файл вирізає roi для кожного виду помилок які ми детектим.
    Типи та розміри roi знаходяться в rois.json. Скоріш за все це тимчасовий підхід, але він потрібний
    для того щоб розділити розробку детекторів та механізм пошуку ROI.
'''



# Шляхи
input_dir = "jpg_discovery"
output_dir = "rois"
roi_json_file = "rois.json"

# Завантаження спільного списку ROI
with open(roi_json_file, "r") as f:
    roi_data = json.load(f)
    rois = roi_data.get("rois", [])

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

        for idx, roi in enumerate(rois):
            roi_type = roi.get("type", "unknown")
            x = roi.get("x", 0)
            y = roi.get("y", 0)
            w = roi.get("w", 0)
            h = roi.get("h", 0)

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
