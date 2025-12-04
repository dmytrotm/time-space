import os
import re
import pandas as pd

# шлях до папки з файлами
dataset_dir = "branchdataset"

# регулярка для розбору назв
pattern = re.compile(
    r"Test_Case_(\d+)_Z([A-Za-z0-9\-_]+)_([A-Za-z0-9\-_]+)_([A-Za-z0-9\-_]+)_ORIENTATION_(\d+)\.png"

)


data = []

for filename in os.listdir(dataset_dir):
    if not filename.endswith(".png"):
        continue

    match = pattern.match(filename)
    if not match:
        print("⚠️ Пропущено (не співпало з шаблоном):", filename)
        continue

    test_case, zone, subzone, iteration, orientation = match.groups()

    test_case = int(test_case)
    orientation = int(orientation)

    # правило маркування
    label = (
        (test_case == 3 and zone == "1" and orientation == 1)
        or (test_case == 3 and zone == "2" and orientation == 4)
        or (test_case == 3 and zone == "2" and orientation == 7)
    )

    data.append({
        "filename": filename,
        "Test_Case": test_case,
        "Zone": zone,
        "Nomer": subzone,
        "Iteration": int(iteration),
        "Orientation": orientation,
        "Label": int(label)
    })

df = pd.DataFrame(data)
df = df.sort_values(by=["Test_Case", "Zone", "Orientation"]).reset_index(drop=True)

csv_path = os.path.join(dataset_dir, "dataset_labels.csv")
df.to_csv(csv_path, index=False)

print("✅ Dataset збережено у:", csv_path)
print(df.head())
