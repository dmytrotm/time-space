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
    """Unit tests для MissingWiresDetector"""
    
    @classmethod
    def setUpClass(cls):
        """Ініціалізація перед всіма тестами"""
        print("\n" + "="*80)
        print("MISSING WIRES DETECTOR - Unit Tests")
        print("="*80)
        
        # Initialize detector
        try:
            cls.detector = MissingWiresDetector()
            print("✓ Detector initialized successfully")
        except FileNotFoundError as e:
            print(f"✗ Failed to initialize detector: {e}")
            raise
        
        # Шлях до dataset (можна змінити на dataset/1 або dataset/2)
        cls.dataset_base = Path("dataset/2")
        
        cls.test_cases = {
            0: {
                'path': cls.dataset_base / "Test_Case_0",
                'expected_missing': ['blue', 'brown', 'black'],
                'description': "БЕЗ дротів (всі відсутні)",
                'allowed_error_rate': 0.0  # No errors allowed
            },
            1: {
                'path': cls.dataset_base / "Test_Case_1",
                'expected_missing': [],
                'description': "З усіма дротами (нічого не відсутнє)",
                'allowed_error_rate': 0.02  # Allow 2% error rate due to image quality
            },
            2: {
                'path': cls.dataset_base / "Test_Case_2",
                'expected_missing': ['blue', 'brown', 'black'],
                'description': "БЕЗ дротів (всі відсутні)",
                'allowed_error_rate': 0.0  # No errors allowed
            }
        }
        
        print(f"Dataset: {cls.dataset_base}")
        print(f"Test Cases: {len(cls.test_cases)}")
        print("="*80 + "\n")
    
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
        if test_case_path.is_dir():
            subdirs = sorted([d for d in test_case_path.iterdir() if d.is_dir()])
            
            for subdir in subdirs:
                # Знайти всі .png та .jpg файли
                images.extend(sorted(subdir.glob("*.png")))
                images.extend(sorted(subdir.glob("*.jpg")))
                images.extend(sorted(subdir.glob("*.jpeg")))
        
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
        allowed_error_rate = tc_info.get('allowed_error_rate', 0.0)
        
        print(f"\n{'='*80}")
        print(f"TEST CASE {tc_num}: {description}")
        print(f"Path: {tc_path}")
        print(f"Expected missing: {expected_missing if expected_missing else 'None'}")
        print(f"{'='*80}")
        
        # Перевірити що папка існує
        if not tc_path.exists():
            print(f"⚠️  SKIPPING: Test case path не існує: {tc_path}")
            self.skipTest(f"Test case path not found: {tc_path}")
            return
        
        # Отримати всі зображення
        images = self._get_all_images_in_test_case(tc_path)
        if len(images) == 0:
            print(f"⚠️  SKIPPING: Жодного зображення не знайдено в {tc_path}")
            self.skipTest(f"No images found in {tc_path}")
            return
        
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
        
        print("\nСтатистика виявлень:")
        for result, count in sorted(results_summary.items()):
            percentage = count / total_images * 100
            print(f"  {result:20s}: {count:3d} ({percentage:5.1f}%)")
        
        # Показати деталі помилок якщо є
        if failed_details:
            print("\nПомилки (перші 10):")
            for detail in failed_details[:10]:
                print(f"  ✗ {detail['image']}")
                print(f"    Expected: {detail['expected']}")
                print(f"    Got:      {detail['actual']}")
        
        print(f"{'-'*80}")
        
        # Calculate error rate
        error_rate = failed_images / total_images if total_images > 0 else 0
        
        # Assert that error rate is within allowed threshold
        self.assertLessEqual(
            error_rate,
            allowed_error_rate,
            f"\n❌ Test Case {tc_num} FAILED: {failed_images}/{total_images} зображень не пройшли тест "
            f"({error_rate*100:.1f}% error rate, allowed: {allowed_error_rate*100:.1f}%)\n"
            f"Очікувалось: {expected_missing}\n"
            f"Детальніше див. вище"
        )
        
        if failed_images == 0:
            print(f"✅ Test Case {tc_num} PASSED: Всі {total_images} зображень коректно оброблені\n")
        else:
            print(f"✅ Test Case {tc_num} PASSED: {passed_images}/{total_images} зображень коректно оброблені "
                  f"({error_rate*100:.1f}% error rate within allowed {allowed_error_rate*100:.1f}%)\n")
    
    def test_case_0_no_wires(self):
        """Test Case 0: БЕЗ дротів (всі відсутні)"""
        self._test_test_case(0)
    
    def test_case_1_all_wires(self):
        """Test Case 1: З усіма дротами"""
        self._test_test_case(1)
    
    def test_case_2_no_wires(self):
        """Test Case 2: БЕЗ дротів (всі відсутні)"""
        self._test_test_case(2)
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        detector = MissingWiresDetector()
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.predictor)
    
    def test_detect_across_rois_returns_dict(self):
        """Test that detect_across_rois returns proper dict structure"""
        # Create a simple test image
        import numpy as np
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.detector.detect_across_rois(test_img)
        
        self.assertIsInstance(result, dict)
        self.assertIn('blue', result)
        self.assertIn('brown', result)
        self.assertIn('black', result)
        self.assertIsInstance(result['blue'], bool)
        self.assertIsInstance(result['brown'], bool)
        self.assertIsInstance(result['black'], bool)
    
    def test_get_missing_wires_returns_list(self):
        """Test that get_missing_wires returns a list"""
        import numpy as np
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.detector.get_missing_wires(test_img)
        
        self.assertIsInstance(result, list)
        # All items should be strings
        for item in result:
            self.assertIsInstance(item, str)
            self.assertIn(item, ['blue', 'brown', 'black'])
    
    def test_set_rois_from_config(self):
        """Test that set_rois_from_config properly updates ROI coordinates"""
        import json
        
        # Load z2 config
        config_path = Path(__file__).parent.parent / "configs" / "rois_z2.json"
        with open(config_path, 'r') as f:
            roi_config = json.load(f)
        
        # Create new detector and set ROIs
        detector = MissingWiresDetector()
        detector.set_rois_from_config(roi_config)
        
        # Verify that cropping coordinates were updated
        self.assertIsNotNone(detector.predictor.cropping_coordinates)
        self.assertIn('blue_brown', detector.predictor.cropping_coordinates)
        self.assertIn('black', detector.predictor.cropping_coordinates)
        
        # Verify coordinates match config
        blue_brown_coords = detector.predictor.cropping_coordinates['blue_brown']
        black_coords = detector.predictor.cropping_coordinates['black']
        
        # Wire 1 (blue_brown)
        self.assertEqual(blue_brown_coords[0], roi_config['wires'][0]['start']['x'])
        self.assertEqual(blue_brown_coords[1], roi_config['wires'][0]['start']['y'])
        self.assertEqual(blue_brown_coords[2], roi_config['wires'][0]['end']['x'])
        self.assertEqual(blue_brown_coords[3], roi_config['wires'][0]['end']['y'])
        
        # Wire 2 (black)
        self.assertEqual(black_coords[0], roi_config['wires'][1]['start']['x'])
        self.assertEqual(black_coords[1], roi_config['wires'][1]['start']['y'])
        self.assertEqual(black_coords[2], roi_config['wires'][1]['end']['x'])
        self.assertEqual(black_coords[3], roi_config['wires'][1]['end']['y'])
    
    @classmethod
    def tearDownClass(cls):
        """Очищення після всіх тестів"""
        print("\n" + "="*80)
        print("ВСІ ТЕСТИ ЗАВЕРШЕНІ")
        print("="*80 + "\n")


def run_quick_summary():
    """
    Швидкий підсумок - прогнати по одному зображенню з кожного тест кейсу
    """
    print("\n" + "="*80)
    print("ШВИДКИЙ ПІДСУМОК: MissingWiresDetector")
    print("="*80 + "\n")
    
    # Import detector
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "missing_wires",
        str(Path(__file__).parent.parent / "detectors" / "missing_wires.py")
    )
    missing_wires_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_wires_module)
    
    try:
        detector = missing_wires_module.MissingWiresDetector()
        print("✓ Detector loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load detector: {e}\n")
        return
    
    dataset_base = Path("dataset/2")
    
    if not dataset_base.exists():
        print(f"⚠️  Dataset not found: {dataset_base}")
        print("Please ensure dataset is available for testing.\n")
        return
    
    test_cases = {
        0: (dataset_base / "Test_Case_0", ['blue', 'brown', 'black'], "БЕЗ дротів"),
        1: (dataset_base / "Test_Case_1", [], "З усіма дротами"),
        2: (dataset_base / "Test_Case_2", ['blue', 'brown', 'black'], "БЕЗ дротів")
    }
    
    for tc_num, (tc_path, expected_missing, description) in test_cases.items():
        if not tc_path.exists():
            print(f"⚠️  TC{tc_num}: Папка не існує - {tc_path}")
            continue
            
        # Знайти перше зображення
        subdirs = sorted([d for d in tc_path.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"⚠️  TC{tc_num}: Папки не знайдені")
            continue
        
        images = sorted(subdirs[0].glob("*.png"))
        if not images:
            images = sorted(subdirs[0].glob("*.jpg"))
        if not images:
            images = sorted(subdirs[0].glob("*.jpeg"))
        
        if not images:
            print(f"⚠️  TC{tc_num}: Зображення не знайдені")
            continue
        
        # Тестувати перше зображення
        try:
            img = cv2.imread(str(images[0]))
            if img is None:
                print(f"⚠️  TC{tc_num}: Не вдалось завантажити зображення")
                continue
                
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
        except Exception as e:
            print(f"✗ TC{tc_num}: Error processing - {e}\n")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Швидкий підсумок
        run_quick_summary()
    else:
        # Повні unit tests
        unittest.main()

