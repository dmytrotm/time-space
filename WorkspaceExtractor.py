import cv2
import numpy as np
from typing import List, Tuple, Optional

from custom_markers_finder import MarkerDetector

import os

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
        
        Returns:
            Масив координат 4 кутових маркерів або None
        """
        if len(corners_list) < 4:
            print("Потрібно мінімум 4 маркери для створення прямокутника")
            return None
        
        # Отримуємо центри всіх маркерів
        centers = []
        for corner in corners_list:
            center = self.detector.compute_center(corner)
            centers.append(center)
        
        centers = np.array(centers)
        
        # Знаходимо кутові точки прямокутника
        # Обчислюємо центр всіх маркерів
        center_point = np.mean(centers, axis=0)
        
        # Класифікуємо маркери за квадрантами відносно центру
        top_left = []
        top_right = []
        bottom_left = []
        bottom_right = []

        for i, center in enumerate(centers):
            if center[0] < center_point[0] and center[1] < center_point[1]:
                top_left.append((i, center))
            elif center[0] >= center_point[0] and center[1] < center_point[1]:
                top_right.append((i, center))
            elif center[0] < center_point[0] and center[1] >= center_point[1]:
                bottom_left.append((i, center))
            else:
                bottom_right.append((i, center))
        
        # Вибираємо найбільш крайні маркери в кожному квадранті
        corner_points = []
        quadrants = [top_left, top_right, bottom_right, bottom_left]
        
        for quadrant in quadrants:
            if not quadrant:
                print("Не вдалося знайти маркери в одному з квадрантів")
                return None
            
            if len(quadrant) == 1:
                corner_points.append(quadrant[0][1])
            else:
                # Вибираємо найбільш віддалений від центру маркер
                distances = [np.linalg.norm(pt[1] - center_point) for pt in quadrant]
                farthest_idx = np.argmax(distances)
                corner_points.append(quadrant[farthest_idx][1])
        
        return np.array(corner_points, dtype=np.float32)
    
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


# Приклад використання
if __name__ == "__main__":
    # Ініціалізуємо детектор маркерів
    detector = MarkerDetector()

    dict_names = ["cust_dictionary4", "cust_dictionary5", "cust_dictionary6", "cust_dictionary8"]
    detector.load_dictionaries("custom_dictionaries.yml", dict_names)
    
    # Ініціалізуємо екстрактор робочої області
    extractor = WorkspaceExtractor(detector)
    
    # Завантажуємо зображення
    image = cv2.imread("jpg_discovery/IMG_1685.jpg")
    
    # Знаходимо маркери
    corners, rejected, ids, _ = detector.detect_all_markers(image)

    output_dir = ""
    
    if ids is not None:
        print(f"Знайдено {len(ids)} маркерів")
        
        # Метод для прямокутної конфігурації маркерів з оригінальним розміром
        workspace = extractor.extract_workspace_from_rectangle(image, corners, ids, None, 
                                                              os.path.join(output_dir, "rectangle_workspace_original.jpg"))
        
        if workspace is not None:
            print(f"Робоча область в оригінальному розмірі збережена як {output_dir}/rectangle_workspace_original.jpg")
            print(f"Розмір вирізаної області: {workspace.shape[1]}x{workspace.shape[0]}")
        
        # Візуалізація автоматично знайдених кутів з інформацією про розмір
        rectangle_viz = extractor.visualize_rectangle_detection(image, corners, ids)
        cv2.imwrite(os.path.join(output_dir, "rectangle_detection_with_size.jpg"), rectangle_viz)
        print(f"Візуалізація прямокутника з розміром збережена як {output_dir}/rectangle_detection_with_size.jpg")
    
    else:
        print("Маркери не знайдені")