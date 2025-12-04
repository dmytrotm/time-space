# import os
# import shutil

# base_dir = "dataset"
# merged_dir = "merged"
# os.makedirs(merged_dir, exist_ok=True)

# # Обходимо всі зони (1 і 2)
# for zone in ["1", "2"]:
#     zone_dir = os.path.join(base_dir, zone)
    
#     # усі тесткейси всередині (Test_Case_0, Test_Case_1, ...)
#     for test_case in os.listdir(zone_dir):
#         test_case_path = os.path.join(zone_dir, test_case)
#         if not os.path.isdir(test_case_path):
#             continue
        
#         # шукаємо всі PNG у підпапках
#         for root, _, files in os.walk(test_case_path):
#             for file in files:
#                 if file.lower().endswith(".png"):
#                     src = os.path.join(root, file)
#                     # додаємо ім'я тесткейсу на початок
#                     new_name = f"{test_case}_{file}"
#                     dst = os.path.join(merged_dir, new_name)
#                     shutil.copy2(src, dst)
#                     print(f"✅ Copied: {src} → {dst}")

# print("🎉 Усі файли скопійовані у 'merged/' з доданим префіксом Test_Case_№")
import os
import re
import shutil

source_base = "branch_wrong_orientation/dataset"
merged_dir = "merged"

os.makedirs(merged_dir, exist_ok=True)

# Регулярка для вилучення Timestamp із назв типу "Frame-1762165562819_(3000, 4000, 3).png"
timestamp_pattern = re.compile(r"Frame-(\d+)_")

for zone in ["1", "2"]:
    zone_dir = os.path.join(source_base, zone)
    if not os.path.exists(zone_dir):
        print(f"⚠️ Пропускаємо {zone_dir} — не знайдено.")
        continue

    for file in os.listdir(zone_dir):
        if not file.lower().endswith(".png"):
            continue

        match = timestamp_pattern.search(file)
        if not match:
            print(f"⚠️ Не вдалося знайти timestamp у {file}")
            continue

        timestamp = match.group(1)
        zone_label = f"Z{zone}"
        new_name = f"Test_Case_3_{zone_label}_0_{timestamp}.png"

        src = os.path.join(zone_dir, file)
        dst = os.path.join(merged_dir, new_name)

        shutil.copy2(src, dst)
        print(f"✅ Copied: {src} → {dst}")

print("🎉 Усі нові кадри додані у 'merged/' з правильними назвами!")
