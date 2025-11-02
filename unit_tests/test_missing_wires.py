#!/usr/bin/env python3
"""
Unit tests для MissingWiresDetector
Тестує детекцію дротів на всіх зображеннях в Test_Case_0, Test_Case_1, Test_Case_2

Expected behavior:
- Test_Case_0: All wires missing (blue, brown, black)
- Test_Case_1: All wires present (no missing wires)
- Test_Case_2: All wires missing (blue, brown, black)
"""

import unittest
import sys
import os
from pathlib import Path
from collections import defaultdict
import importlib.util

# Додати шлях до проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

# Import directly from file to avoid loading other detectors
spec = importlib.util.spec_from_file_location(
    "missing_wires",
    str(Path(__file__).parent.parent / "detectors" / "missing_wires.py")
)
missing_wires_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(missing_wires_module)
MissingWiresDetector = missing_wires_module.MissingWiresDetector


class TestMissingWiresDetector(unittest.TestCase):
    """Unit tests для детектора відсутніх дротів"""
    
    @classmethod
    def setUpClass(cls):
        """Ініціалізація перед всіма тестами"""
        cls.detector = MissingWiresDetector()
        
        # Шлях до dataset (можна змінити на dataset/1 або dataset/2)
        cls.dataset_base = Path("dataset/2")
        
        cls.test_cases = {
            0: {
                'path': cls.dataset_base / "Test_Case_0",
                'expected_missing': ['blue', 'brown', 'black'],
                'description': "БЕЗ дротів (всі відсутні)"
            },
            1: {
                'path': cls.dataset_base / "Test_Case_1",
                'expected_missing': [],
                'description': "З усіма дротами (нічого не відсутнє)"
            },
            2: {
                'path': cls.dataset_base / "Test_Case_2",
                'expected_missing': ['blue', 'brown', 'black'],
                'description': "БЕЗ дротів (всі відсутні)"
            }
        }
        
        print(f"\n{'='*80}")
        print(f"MISSING WIRES DETECTOR - Unit Tests")
        print(f"{'='*80}")
        print(f"Dataset: {cls.dataset_base}")
        print(f"Test Cases: {len(cls.test_cases)}")
        print(f"{'='*80}\n")
    
    def _get_all_images_in_test_case(self, test_case_path):
        """
        Отримати всі зображення з усіх підпапок тест кейсу.
        
        Args:
            test_case_path: Path до Test_Case_X папки
            
        Returns:
            list: Список Path об'єктів до всіх зображень
        """
        images = []
        
        # Знайти всі підпапки (1, 2, 3, 4, 5, etc.)
        subdirs = sorted([d for d in test_case_path.iterdir() if d.is_dir()])
        
        for subdir in subdirs:
            # Знайти всі .png та .jpg файли
            images.extend(sorted(subdir.glob("*.png")))
            images.extend(sorted(subdir.glob("*.jpg")))
        
        return images
    
    def _test_test_case(self, tc_num):
        """
        Тестує всі зображення в тест кейсі.
        
        Args:
            tc_num: Номер тест кейсу (0, 1, 2)
        """
        tc_info = self.test_cases[tc_num]
        tc_path = tc_info['path']
        expected_missing = sorted(tc_info['expected_missing'])
        description = tc_info['description']
        
        print(f"\n{'='*80}")
        print(f"TEST CASE {tc_num}: {description}")
        print(f"Path: {tc_path}")
        print(f"Expected missing: {expected_missing if expected_missing else 'None'}")
        print(f"{'='*80}")
        
        # Перевірити що папка існує
        self.assertTrue(tc_path.exists(), f"Test case path не існує: {tc_path}")
        
        # Отримати всі зображення
        images = self._get_all_images_in_test_case(tc_path)
        self.assertGreater(len(images), 0, f"Жодного зображення не знайдено в {tc_path}")
        
        print(f"\nЗнайдено зображень: {len(images)}")
        
        # Лічильники для статистики
        total_images = len(images)
        passed_images = 0
        failed_images = 0
        results_summary = defaultdict(int)
        failed_details = []
        
        # Тестувати кожне зображення
        for i, image_path in enumerate(images, 1):
            img = cv2.imread(str(image_path))
            self.assertIsNotNone(img, f"Не вдалось завантажити зображення: {image_path}")
            
            # Виявити відсутні дроти
            missing_wires = sorted(self.detector.get_missing_wires(img, use_rois=True))
            
            # Перевірити результат
            is_correct = missing_wires == expected_missing
            
            if is_correct:
                passed_images += 1
                status = "✓"
            else:
                failed_images += 1
                status = "✗"
                failed_details.append({
                    'image': image_path.name,
                    'expected': expected_missing,
                    'actual': missing_wires
                })
            
            # Записати результат для статистики
            results_key = ','.join(missing_wires) if missing_wires else 'none'
            results_summary[results_key] += 1
            
            # Вивести прогрес кожні 10 зображень або якщо помилка
            if i % 10 == 0 or not is_correct or i == total_images:
                print(f"  [{i:3d}/{total_images}] {status} {image_path.name:30s} | "
                      f"Missing: {missing_wires if missing_wires else 'none'}")
        
        # Підсумок
        print(f"\n{'-'*80}")
        print(f"РЕЗУЛЬТАТИ Test Case {tc_num}:")
        print(f"  Всього зображень:  {total_images}")
        print(f"  Пройшло:           {passed_images} ({passed_images/total_images*100:.1f}%)")
        print(f"  Не пройшло:        {failed_images} ({failed_images/total_images*100:.1f}%)")
        
        print(f"\nСтатистика виявлень:")
        for result, count in sorted(results_summary.items()):
            percentage = count / total_images * 100
            print(f"  {result:20s}: {count:3d} ({percentage:5.1f}%)")
        
        # Показати деталі помилок якщо є
        if failed_details:
            print(f"\nПомилки (перші 10):")
            for detail in failed_details[:10]:
                print(f"  ✗ {detail['image']}")
                print(f"    Expected: {detail['expected']}")
                print(f"    Got:      {detail['actual']}")
        
        print(f"{'-'*80}")
        
        # Assert що всі зображення пройшли тест
        self.assertEqual(
            failed_images, 
            0, 
            f"\n❌ Test Case {tc_num} FAILED: {failed_images}/{total_images} зображень не пройшли тест\n"
            f"Очікувалось: {expected_missing}\n"
            f"Детальніше див. вище"
        )
        
        print(f"✅ Test Case {tc_num} PASSED: Всі {total_images} зображень коректно оброблені\n")
    
    def test_case_0_no_wires(self):
        """Test Case 0: БЕЗ дротів (всі відсутні)"""
        self._test_test_case(0)
    
    def test_case_1_all_wires(self):
        """Test Case 1: З усіма дротами"""
        self._test_test_case(1)
    
    def test_case_2_no_wires(self):
        """Test Case 2: БЕЗ дротів (всі відсутні)"""
        self._test_test_case(2)
    
    @classmethod
    def tearDownClass(cls):
        """Очищення після всіх тестів"""
        print(f"\n{'='*80}")
        print(f"ВСІ ТЕСТИ ЗАВЕРШЕНІ")
        print(f"{'='*80}\n")


def run_quick_summary():
    """
    Швидкий підсумок - прогнати по одному зображенню з кожного тест кейсу
    """
    print(f"\n{'='*80}")
    print("ШВИДКИЙ ПІДСУМОК: MissingWiresDetector")
    print(f"{'='*80}\n")
    
    # Import detector
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "missing_wires",
        str(Path(__file__).parent.parent / "detectors" / "missing_wires.py")
    )
    missing_wires_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_wires_module)
    
    detector = missing_wires_module.MissingWiresDetector()
    dataset_base = Path("dataset/2")
    
    test_cases = {
        0: (dataset_base / "Test_Case_0", ['blue', 'brown', 'black'], "БЕЗ дротів"),
        1: (dataset_base / "Test_Case_1", [], "З усіма дротами"),
        2: (dataset_base / "Test_Case_2", ['blue', 'brown', 'black'], "БЕЗ дротів")
    }
    
    for tc_num, (tc_path, expected_missing, description) in test_cases.items():
        # Знайти перше зображення
        subdirs = sorted([d for d in tc_path.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"⚠️  TC{tc_num}: Папки не знайдені")
            continue
        
        images = sorted(subdirs[0].glob("*.png"))
        if not images:
            images = sorted(subdirs[0].glob("*.jpg"))
        
        if not images:
            print(f"⚠️  TC{tc_num}: Зображення не знайдені")
            continue
        
        # Тестувати перше зображення
        img = cv2.imread(str(images[0]))
        detected = detector.detect_across_rois(img)
        missing = detector.get_missing_wires(img)
        
        # Статус дротів
        status = {
            'blue': '✓' if detected['blue'] else '✗',
            'brown': '✓' if detected['brown'] else '✗',
            'black': '✓' if detected['black'] else '✗'
        }
        
        is_correct = sorted(missing) == sorted(expected_missing)
        result_icon = "✅" if is_correct else "❌"
        
        print(f"TC{tc_num} ({description}): {result_icon}")
        print(f"  File: {images[0].name}")
        print(f"  Blue: {status['blue']} | Brown: {status['brown']} | Black: {status['black']}")
        print(f"  Missing: {missing if missing else 'none'}")
        print(f"  Expected: {expected_missing if expected_missing else 'none'}")
        print()
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Швидкий підсумок
        run_quick_summary()
    else:
        # Повні unit tests
        unittest.main()
