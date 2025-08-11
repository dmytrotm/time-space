import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import glob
from pathlib import Path

from custom_markers_finder import MarkerDetector

class WorkspaceExtractor:
    def __init__(self, marker_detector):
        self.detector = marker_detector
        
    def find_workspace_corners(self, corners_list: List, ids_list: np.ndarray, 
                              corner_marker_ids: List[int]) -> Optional[np.ndarray]:
        """
        Знаходить кутові точки робочої області за заданими ID маркерів
        
        Args:
            corners_list: Список кутів всіх знайдених маркерів
            ids_list: Масив ID всіх знайдених маркерів
            corner_marker_ids: Список ID маркерів, які визначають кути робочої області
                              [top_left_id, top_right_id, bottom_right_id, bottom_left_id]
        
        Returns:
            Масив координат кутів робочої області або None
        """
        if len(corner_marker_ids) != 4:
            raise ValueError("Потрібно вказати рівно 4 ID маркера для кутів")
        
        workspace_corners = []
        found_ids = []
        
        for target_id in corner_marker_ids:
            found = False
            for i, marker_id in enumerate(ids_list):
                true_id = marker_id[0] % 10000  # Видаляємо словник і border info
                if true_id == target_id:
                    # Беремо центр маркера як точку кута
                    center = self.detector.compute_center(corners_list[i])
                    workspace_corners.append(center)
                    found_ids.append(true_id)
                    found = True
                    break
            
            if not found:
                print(f"Маркер з ID {target_id} не знайдений")
                return None
        
        if len(workspace_corners) == 4:
            return np.array(workspace_corners, dtype=np.float32)
        else:
            return None
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Упорядковує точки у правильному порядку: top-left, top-right, bottom-right, bottom-left
        """
        # Сортуємо точки за сумою координат
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left матиме найменшу суму, bottom-right - найбільшу
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Top-right матиме найменшу різницю, bottom-left - найбільшу
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def calculate_original_size(self, corners: np.ndarray) -> Tuple[int, int]:
        """
        Обчислює оригінальний розмір прямокутника за координатами кутів
        
        Args:
            corners: Упорядковані координати кутів [top_left, top_right, bottom_right, bottom_left]
        
        Returns:
            Tuple (width, height) оригінального розміру
        """
        # Обчислюємо відстані між кутами
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        
        # Використовуємо середні значення для більшої точності
        width = int((top_width + bottom_width) / 2)
        height = int((left_height + right_height) / 2)
        
        return width, height
    
    def extract_workspace(self, image: np.ndarray, corners_list: List, ids_list: np.ndarray,
                         corner_marker_ids: List[int], output_size: Optional[Tuple[int, int]] = None,
                         save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Вирізає та вирівнює робочу область
        
        Args:
            image: Вхідне зображення
            corners_list: Список кутів маркерів
            ids_list: Масив ID маркерів
            corner_marker_ids: ID маркерів кутів [top_left, top_right, bottom_right, bottom_left]
            output_size: Розмір вихідного зображення (width, height). Якщо None, використовується оригінальний розмір
        
        Returns:
            Вирізане та вирівняне зображення робочої області
        """
        # Знаходимо кути робочої області
        workspace_corners = self.find_workspace_corners(corners_list, ids_list, corner_marker_ids)
        
        if workspace_corners is None:
            return None
        
        # Упорядковуємо точки
        ordered_corners = self.order_points(workspace_corners)
        
        # Визначаємо розмір виходу
        if output_size is None:
            output_size = self.calculate_original_size(ordered_corners)
        
        # Створюємо цільові точки для перспективного перетворення
        dst_points = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1]
        ], dtype=np.float32)
        
        # Обчислюємо матрицю перспективного перетворення
        perspective_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        
        # Застосовуємо перспективне перетворення
        warped = cv2.warpPerspective(image, perspective_matrix, output_size)
        
        # Зберігаємо, якщо вказано шлях
        if save_path:
            cv2.imwrite(save_path, warped)
        
        return warped
    
    def extract_workspace_auto(self, image: np.ndarray, corners_list: List, ids_list: np.ndarray,
                              margin: int = 50, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Автоматично вирізає робочу область, використовуючи всі знайдені маркери
        
        Args:
            image: Вхідне зображення
            corners_list: Список кутів маркерів
            ids_list: Масив ID маркерів
            margin: Відступ від крайніх маркерів
        
        Returns:
            Вирізане зображення робочої області
        """
        if len(corners_list) == 0:
            return None
        
        # Знаходимо всі центри маркерів
        centers = []
        for corner in corners_list:
            center = self.detector.compute_center(corner)
            centers.append(center)
        
        centers = np.array(centers)
        
        # Знаходимо межі робочої області
        min_x = np.min(centers[:, 0]) - margin
        max_x = np.max(centers[:, 0]) + margin
        min_y = np.min(centers[:, 1]) - margin
        max_y = np.max(centers[:, 1]) + margin
        
        # Обмежуємо координати межами зображення
        min_x = max(0, int(min_x))
        max_x = min(image.shape[1], int(max_x))
        min_y = max(0, int(min_y))
        max_y = min(image.shape[0], int(max_y))
        
        # Вирізаємо область
        cropped = image[min_y:max_y, min_x:max_x]
        
        # Зберігаємо, якщо вказано шлях
        if save_path:
            cv2.imwrite(save_path, cropped)
        
        return cropped
    
    def find_rectangle_corners(self, corners_list: List, ids_list: np.ndarray) -> Optional[np.ndarray]:
        """
        Автоматично знаходить 4 кутові маркери з прямокутної конфігурації
        """
        if len(corners_list) < 4:
            print("Потрібно мінімум 4 маркери для створення прямокутника")
            return None
        
        # Отримуємо центри та true_ids
        id_to_center = {}
        for i, marker_id in enumerate(ids_list):
            true_id = marker_id[0] % 10000
            center = self.detector.compute_center(corners_list[i])
            id_to_center[true_id] = center
        
        # Відомі ID
        corner_ids = {
            'UL': 0,    # Top-left
            'UR': 2001, # Top-right
            'LR': 2002, # Bottom-right
            'LL': 3000  # Bottom-left
        }
        middle_ids = {
            'ML': 2000, # Middle-left
            'MR': 1000  # Middle-right
        }
        
        # Отримуємо позиції
        positions = {}
        for pos, tid in corner_ids.items():
            if tid in id_to_center:
                positions[pos] = id_to_center[tid]
        for pos, tid in middle_ids.items():
            if tid in id_to_center:
                positions[pos] = id_to_center[tid]
        
        # Естимуємо missing corners (з пріоритетом на кращі комбінації)
        estimated = []
        for pos in list(corner_ids.keys()):  # Копіюємо, бо можемо додавати
            if pos not in positions:
                if pos == 'UL':
                    # Пріоритет: LL + UR (ліва x + верхня y)
                    if 'LL' in positions and 'UR' in positions:
                        ul_x = positions['LL'][0]  # Ліва x з LL
                        ul_y = positions['UR'][1]  # Верхня y з UR
                        positions['UL'] = np.array([ul_x, ul_y])
                        estimated.append('UL (x from LL, y from UR)')
                    # Альтернатива: ML + UR (ліва x + верхня y)
                    elif 'ML' in positions and 'UR' in positions:
                        ul_x = positions['ML'][0]
                        ul_y = positions['UR'][1]
                        positions['UL'] = np.array([ul_x, ul_y])
                        estimated.append('UL (x from ML, y from UR)')
                
                elif pos == 'UR':
                    # Пріоритет: LR + UL (права x + верхня y)
                    if 'LR' in positions and 'UL' in positions:
                        ur_x = positions['LR'][0]
                        ur_y = positions['UL'][1]
                        positions['UR'] = np.array([ur_x, ur_y])
                        estimated.append('UR (x from LR, y from UL)')
                    # Альтернатива: MR + UL
                    elif 'MR' in positions and 'UL' in positions:
                        ur_x = positions['MR'][0]
                        ur_y = positions['UL'][1]
                        positions['UR'] = np.array([ur_x, ur_y])
                        estimated.append('UR (x from MR, y from UL)')
                
                elif pos == 'LR':
                    # Пріоритет: UR + LL (права x + нижня y)
                    if 'UR' in positions and 'LL' in positions:
                        lr_x = positions['UR'][0]
                        lr_y = positions['LL'][1]
                        positions['LR'] = np.array([lr_x, lr_y])
                        estimated.append('LR (x from UR, y from LL)')
                    # Альтернатива: MR + LL
                    elif 'MR' in positions and 'LL' in positions:
                        lr_x = positions['MR'][0]
                        lr_y = positions['LL'][1]
                        positions['LR'] = np.array([lr_x, lr_y])
                        estimated.append('LR (x from MR, y from LL)')
                
                elif pos == 'LL':
                    # Пріоритет: UL + LR (ліва x + нижня y)
                    if 'UL' in positions and 'LR' in positions:
                        ll_x = positions['UL'][0]
                        ll_y = positions['LR'][1]
                        positions['LL'] = np.array([ll_x, ll_y])
                        estimated.append('LL (x from UL, y from LR)')
                    # Альтернатива: екстраполяція з UL + ML (якщо немає нижньої info)
                    elif 'UL' in positions and 'ML' in positions:
                        dx = positions['ML'][0] - positions['UL'][0]  # Зсув x (зазвичай ~0)
                        dy = positions['ML'][1] - positions['UL'][1]  # Половина висоти
                        ll_x = positions['ML'][0] + dx
                        ll_y = positions['ML'][1] + dy
                        positions['LL'] = np.array([ll_x, ll_y])
                        estimated.append('LL (extrapolated from UL and ML)')
        
        # Перевіряємо чи всі кути є
        missing_after = [p for p in corner_ids if p not in positions]
        if missing_after:
            print(f"Не вдалося естимувати missing corners: {missing_after}")
            return None
        
        if estimated:
            print(f"Естимовано позиції для: {estimated}")
        
        # Додаткова перевірка логічності (основні нерівності для прямокутника)
        if positions['UL'][0] >= positions['UR'][0] or positions['UL'][1] >= positions['LL'][1]:
            print("Естимовані кути нелогічні (порушення порядку x/y), ігноруємо естимацію")
            return None
        
        # Збираємо в масив в порядку UL, UR, LR, LL
        workspace_corners = [
            positions['UL'],
            positions['UR'],
            positions['LR'],
            positions['LL']
        ]
        return np.array(workspace_corners, dtype=np.float32)
    
    def extract_workspace_from_rectangle(self, image: np.ndarray, corners_list: List, ids_list: np.ndarray,
                                        output_size: Optional[Tuple[int, int]] = None, 
                                        save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Вирізає робочу область автоматично з прямокутної конфігурації маркерів
        
        Args:
            image: Вхідне зображення
            corners_list: Список кутів маркерів
            ids_list: Масив ID маркерів
            output_size: Розмір вихідного зображення (width, height). Якщо None, використовується оригінальний розмір
            save_path: Шлях для збереження (опційно)
        
        Returns:
            Вирізане та вирівняне зображення робочої області
        """
        # Автоматично знаходимо кутові маркери
        workspace_corners = self.find_rectangle_corners(corners_list, ids_list)
        
        if workspace_corners is None:
            return None
        
        # Упорядковуємо точки
        ordered_corners = self.order_points(workspace_corners)
        
        # Визначаємо розмір виходу
        if output_size is None:
            output_size = self.calculate_original_size(ordered_corners)
            print(f"Автоматично визначений розмір: {output_size[0]}x{output_size[1]}")
        
        # Створюємо цільові точки для перспективного перетворення
        dst_points = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1]
        ], dtype=np.float32)
        
        # Обчислюємо матрицю перспективного перетворення
        perspective_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        
        # Застосовуємо перспективне перетворення
        warped = cv2.warpPerspective(image, perspective_matrix, output_size)
        
        # Зберігаємо, якщо вказано шлях
        if save_path:
            cv2.imwrite(save_path, warped)
        
        return warped
    
    def visualize_rectangle_detection(self, image: np.ndarray, corners_list: List, ids_list: np.ndarray) -> np.ndarray:
        """
        Візуалізує автоматично знайдені кутові маркери
        """
        output = image.copy()
        
        # Знаходимо кутові маркери
        workspace_corners = self.find_rectangle_corners(corners_list, ids_list)
        
        if workspace_corners is None:
            return output
        
        # Малюємо всі маркери сірим
        for i, corner in enumerate(corners_list):
            center = self.detector.compute_center(corner).astype(int)
            cv2.circle(output, tuple(center), 15, (128, 128, 128), -1)
            cv2.putText(output, f"ID:{ids_list[i][0]}", (center[0] + 20, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        
        # Малюємо кутові маркери зеленим
        ordered_corners = self.order_points(workspace_corners)
        pts = ordered_corners.astype(np.int32)
        
        # Малюємо контур робочої області
        cv2.polylines(output, [pts], True, (0, 255, 0), 3)
        
        # Нумеруємо кути
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (pt, label) in enumerate(zip(pts, labels)):
            cv2.circle(output, tuple(pt), 10, (0, 255, 0), -1)
            cv2.putText(output, label, (pt[0] + 15, pt[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Додаємо інформацію про розмір
        original_size = self.calculate_original_size(ordered_corners)
        cv2.putText(output, f"Size: {original_size[0]}x{original_size[1]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output

    def process_directory(self, input_dir: str, output_base_dir: str = "output"):
        """
        Обробляє всі зображення в директорії та зберігає результати в структурованих папках
        
        Args:
            input_dir: Шлях до директорії з вхідними зображеннями
            output_base_dir: Базова директорія для збереження результатів
        """
        # Створюємо базову директорію для виходу
        output_base_path = Path(output_base_dir)
        output_base_path.mkdir(exist_ok=True)
        
        # Створюємо підпапки для різних типів результатів
        workspace_dir = output_base_path / "workspace_extracted"
        visualization_dir = output_base_path / "rectangle_detection"
        failed_dir = output_base_path / "failed_processing"
        
        workspace_dir.mkdir(exist_ok=True)
        visualization_dir.mkdir(exist_ok=True)
        failed_dir.mkdir(exist_ok=True)
        
        # Підтримувані формати зображень
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        # Знаходимо всі файли зображень
        image_files = []
        input_path = Path(input_dir)
        
        if input_path.is_file():
            # Якщо передано конкретний файл
            image_files = [input_path]
        else:
            # Якщо передано директорію
            for ext in image_extensions:
                image_files.extend(input_path.glob(ext))
                image_files.extend(input_path.glob(ext.upper()))
        
        if not image_files:
            print(f"Не знайдено зображень в {input_dir}")
            return
        
        print(f"Знайдено {len(image_files)} зображень для обробки")
        
        # Лічильники успішності
        success_count = 0
        failed_count = 0
        
        # Обробляємо кожне зображення
        for img_path in image_files:
            print(f"\nОбробка: {img_path.name}")
            
            try:
                # Завантажуємо зображення
                image = cv2.imread(str(img_path))
                
                if image is None:
                    print(f"Не вдалося завантажити {img_path.name}")
                    failed_count += 1
                    continue
                
                # Знаходимо маркери
                corners, rejected, ids, _ = self.detector.detect_all_markers(image)
                
                if ids is not None and len(ids) >= 4:
                    print(f"Знайдено {len(ids)} маркерів")
                    
                    # Генеруємо імена файлів на основі оригінального імені
                    base_name = img_path.stem
                    
                    # Вирізаємо робочу область
                    workspace_save_path = workspace_dir / f"{base_name}_workspace.jpg"
                    workspace = self.extract_workspace_from_rectangle(
                        image, corners, ids, None, str(workspace_save_path)
                    )
                    
                    # Створюємо візуалізацію
                    viz_save_path = visualization_dir / f"{base_name}_detection.jpg"
                    rectangle_viz = self.visualize_rectangle_detection(image, corners, ids)
                    cv2.imwrite(str(viz_save_path), rectangle_viz)
                    
                    if workspace is not None:
                        print(f"✓ Робоча область збережена: {workspace_save_path}")
                        print(f"✓ Візуалізація збережена: {viz_save_path}")
                        print(f"  Розмір вирізаної області: {workspace.shape[1]}x{workspace.shape[0]}")
                        success_count += 1
                    else:
                        print(f"✗ Не вдалося вирізати робочу область")
                        # Копіюємо оригінал в папку невдалих
                        failed_save_path = failed_dir / f"{base_name}_no_workspace.jpg"
                        cv2.imwrite(str(failed_save_path), image)
                        failed_count += 1
                else:
                    print(f"✗ Недостатньо маркерів знайдено (потрібно мінімум 4)")
                    # Копіюємо оригінал в папку невдалих
                    failed_save_path = failed_dir / f"{base_name}_insufficient_markers.jpg"
                    cv2.imwrite(str(failed_save_path), image)
                    failed_count += 1
                    
            except Exception as e:
                print(f"✗ Помилка при обробці {img_path.name}: {str(e)}")
                failed_count += 1
        
        # Виводимо підсумки
        print(f"\n{'='*50}")
        print(f"ПІДСУМКИ ОБРОБКИ")
        print(f"{'='*50}")
        print(f"Успішно оброблено: {success_count}")
        print(f"Невдалих спроб: {failed_count}")
        print(f"Загалом файлів: {len(image_files)}")
        print(f"\nРезультати збережено в:")
        print(f"  Робочі області: {workspace_dir}")
        print(f"  Візуалізації: {visualization_dir}")
        if failed_count > 0:
            print(f"  Невдалі файли: {failed_dir}")


# Приклад використання
if __name__ == "__main__":
    # Ініціалізуємо детектор маркерів
    detector = MarkerDetector()

    dict_names = ["cust_dictionary4", "cust_dictionary5", "cust_dictionary6", "cust_dictionary8"]
    detector.load_dictionaries("custom_dictionaries.yml", dict_names)
    
    # Ініціалізуємо екстрактор робочої області
    extractor = WorkspaceExtractor(detector)
    
    # Варіанти використання:
    
    # 1. Обробка одного файлу
    # extractor.process_directory("workplace/rectangle_workspace_original.jpg", "project_output")
    
    # 2. Обробка всієї директорії
    extractor.process_directory("jpg_discovery", "project_output")
    
    # 3. Можна також використовувати старий спосіб для одного зображення:
    """
    image = cv2.imread("jpg_discovery/IMG_1685.jpg")
    corners, rejected, ids, _ = detector.detect_all_markers(image)
    
    if ids is not None:
        print(f"Знайдено {len(ids)} маркерів")
        
        # Створюємо папку output/workspace в проекті
        output_dir = Path("output/workspace")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Зберігаємо в проектну папку
        workspace = extractor.extract_workspace_from_rectangle(
            image, corners, ids, None, 
            str(output_dir / "rectangle_workspace_original.jpg")
        )
        
        if workspace is not None:
            print(f"Робоча область збережена в {output_dir}/rectangle_workspace_original.jpg")
            print(f"Розмір вирізаної області: {workspace.shape[1]}x{workspace.shape[0]}")
        
        # Візуалізація
        rectangle_viz = extractor.visualize_rectangle_detection(image, corners, ids)
        cv2.imwrite(str(output_dir / "rectangle_detection_with_size.jpg"), rectangle_viz)
        print(f"Візуалізація збережена в {output_dir}/rectangle_detection_with_size.jpg")
    else:
        print("Маркери не знайдені")
    """