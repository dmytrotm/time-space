import os
import cv2
import json
import wires_mask


def cut_the_crops(roi,units:str,img:cv2.UMat):
    img_height, img_width = img.shape[:2]

    x = roi.get("x", 0)
    y = roi.get("y", 0)
    w = roi.get("w", 0)
    h = roi.get("h", 0)

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

    crop = img[y:y+h, x:x+w]
    return crop
    


# Шляхи
def cut_rois(roi_data,input_dir,output_dir):
    
    if not os.path.isdir(input_dir):
        print(f"Input directory {input_dir} is not exist")
        return

    units = roi_data.get("units", "absolute").lower()  
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

            for idx, roi in enumerate(rois):
                crop = cut_the_crops(roi,units,img)
                if crop.size == 0:
                    print(f"[!] Порожнє ROI у {filename}, тип {roi_type}, пропущено.")
                    continue
                roi_type = roi.get("type", "unknown")
                
                # Створити папку для типу
                type_folder = os.path.join(output_dir, roi_type)
                os.makedirs(type_folder, exist_ok=True)

                # Ім’я для ROI-зображення
                roi_filename = f"{name}.jpg"
                roi_path = os.path.join(type_folder, roi_filename)

                # Збереження ROI
                cv2.imwrite(roi_path, crop)

    print("✅ Усі ROI успішно вирізано і збережено.")

def make_masks(input_dir,output_dir,extractor:wires_mask.WiresMaskExtractorInterface):
    
    if not os.path.isdir(input_dir):
        print(f"Input directory {input_dir} is not exist")
        return
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            name, _ = os.path.splitext(filename)
            img_path = os.path.join(input_dir, filename)

            img = cv2.imread(img_path)
            mask = extractor.get_wires_mask(img)
            result = cv2.bitwise_and(img,img,mask=mask)
            path = os.path.join(output_dir,filename)
            cv2.imwrite(path,result)
    print("✅ Усі Маски успішно накладені та збережені")



if __name__ == "__main__":
    input_dir = "jpg_discovery"
    output_dir = "rois"
    roi_json_file = "rois.json"

    # # Завантаження списку ROI
    with open(roi_json_file, "r") as f:
        roi_data = json.load(f)
    gen_roi = roi_data.get("general_rois",{})
    cut_rois(gen_roi,input_dir,output_dir)

    input_dir = "rois/Tailing missing"
    output_dir = "masks/Tailing missing"
    extractor = wires_mask.SaturationWiresMaskExtractor()
    make_masks(input_dir,output_dir,extractor)

    input_dir = "masks/Tailing missing"
    output_dir = "tapes"
    tape_roi = roi_data.get("tape_rois",{})
    cut_rois(tape_roi,input_dir,output_dir)
