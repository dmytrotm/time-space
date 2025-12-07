#!/usr/bin/env python3
"""
Inference скрипт для ResNet18 на Hailo AI HAT+
Підтримує: зображення, відео, real-time камеру
ResNet18 (224x224) з sigmoid виходом
"""

import numpy as np
import cv2
from pathlib import Path
import time
import argparse
from typing import Tuple

try:
    from hailo_platform import (VDevice, HailoStreamInterface, InferVStreams, 
                                ConfigureParams, InputVStreamParams, OutputVStreamParams,
                                FormatType)
except ImportError:
    print("❌ Hailo Platform SDK не встановлено!")
    print("Встанови на Raspberry Pi: sudo apt install hailo-all")
    exit(1)


class ResNet18Inference:
    """ResNet18 inference на Hailo з точним preprocessing"""
    
    def __init__(self, hef_path: str, threshold: float = 0.5, img_size: int = 224):
        self.hef_path = hef_path
        self.threshold = threshold
        self.img_size = img_size
        
        # ImageNet normalization (як у твоєму transforms.Normalize)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.device = None
        self.network_group = None
        
    def __enter__(self):
        """Ініціалізація Hailo пристрою"""
        print(f"🔌 Підключення до Hailo пристрою...")
        self.device = VDevice()
        
        print(f"📦 Завантаження моделі: {self.hef_path}")
        with open(self.hef_path, 'rb') as f:
            hef_data = f.read()
        
        network_groups = self.device.configure(
            ConfigureParams.create_from_hef(hef_data, interface=HailoStreamInterface.PCIe)
        )
        self.network_group = network_groups[0]
        
        print(f"✅ ResNet18 завантажено!")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Очищення ресурсів"""
        if self.network_group:
            self.network_group.release()
        if self.device:
            self.device.release()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing як у твоєму PyTorch коді:
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        Args:
            image: BGR зображення від OpenCV
        Returns:
            Preprocessed batch (1, 3, 224, 224)
        """
        # Resize (еквівалент transforms.Resize)
        resized = cv2.resize(image, (self.img_size, self.img_size), 
                            interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # ToTensor: конвертувати в float32 і ділити на 255
        tensor = rgb.astype(np.float32) / 255.0
        
        # Normalize з ImageNet statistics
        normalized = (tensor - self.mean) / self.std
        
        # HWC -> CHW (як PyTorch tensor)
        chw = np.transpose(normalized, (2, 0, 1))
        
        # Batch dimension
        batch = np.expand_dims(chw, axis=0)  # (1, 3, 224, 224)
        
        return batch
    
    def postprocess(self, output: np.ndarray) -> dict:
        """
        Постпроцесинг sigmoid виходу
        
        Args:
            output: Raw sigmoid output (1, 1)
        Returns:
            {'score': float, 'prediction': bool, 'label': str}
        """
        # Sigmoid вихід - один скаляр
        score = float(output.flatten()[0])
        prediction = score >= self.threshold
        
        return {
            'score': score,
            'prediction': prediction,
            'label': 'positive' if prediction else 'negative'
        }
    
    def infer(self, image: np.ndarray) -> dict:
        """
        Запустити inference на зображенні
        
        Args:
            image: BGR зображення від OpenCV
        Returns:
            {'score': float, 'prediction': bool, 'label': str}
        """
        # Preprocessing
        input_data = self.preprocess(image)
        
        # Inference на Hailo
        with InferVStreams(
            self.network_group,
            InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            ),
            OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
        ) as infer_pipeline:
            
            input_dict = {list(infer_pipeline.input_vstreams.keys())[0]: input_data}
            output_dict = infer_pipeline.infer(input_dict)
            output = list(output_dict.values())[0]
        
        # Postprocessing
        result = self.postprocess(output)
        
        return result


def visualize_result(image: np.ndarray, result: dict) -> np.ndarray:
    """Візуалізація результату класифікації"""
    vis_image = image.copy()
    h, w = vis_image.shape[:2]
    
    # Колір залежно від результату
    color = (0, 255, 0) if result['prediction'] else (0, 0, 255)
    label = f"{result['label'].upper()}: {result['score']:.3f}"
    
    # Background box для тексту
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
    )
    
    cv2.rectangle(vis_image, (10, 10), (20 + text_w, 30 + text_h), color, -1)
    cv2.putText(vis_image, label, (15, 25 + text_h),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Додаткова інформація
    info = f"Threshold: {result.get('threshold', 0.5):.2f}"
    cv2.putText(vis_image, info, (15, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_image


def process_image(model: ResNet18Inference, image_path: str, 
                 output_path: str = None, show: bool = False):
    """Обробка одного зображення"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не вдалося прочитати: {image_path}")
        return None
    
    print(f"📸 Зображення: {image.shape[1]}x{image.shape[0]}")
    
    # Inference
    start_time = time.time()
    result = model.infer(image)
    inference_time = time.time() - start_time
    
    print(f"⏱️  Час inference: {inference_time*1000:.2f} ms")
    print(f"📊 Результат: {result['label']} (score: {result['score']:.3f})")
    
    # Візуалізація
    vis_image = visualize_result(image, result)
    
    # Збереження
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"💾 Збережено: {output_path}")
    
    # Показати
    if show:
        cv2.imshow('ResNet18 Classification', vis_image)
        print("Натисни будь-яку клавішу для закриття...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def process_batch(model: ResNet18Inference, image_dir: str, 
                 output_dir: str = None):
    """Batch обробка папки зображень"""
    image_dir = Path(image_dir)
    
    # Знайти всі зображення
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
        images.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not images:
        print(f"❌ Зображення не знайдено в {image_dir}")
        return
    
    print(f"📁 Знайдено {len(images)} зображень")
    
    # Створити output директорію
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Статистика
    results = {'positive': 0, 'negative': 0}
    total_time = 0
    
    print(f"\n{'='*60}")
    print("Початок batch обробки...")
    print(f"{'='*60}\n")
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}...", end=' ')
        
        image = cv2.imread(str(img_path))
        if image is None:
            print("❌ Помилка читання")
            continue
        
        # Inference
        start_time = time.time()
        result = model.infer(image)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Статистика
        results[result['label']] += 1
        
        print(f"{result['label']:8s} ({result['score']:.3f}) - {inference_time*1000:.1f}ms")
        
        # Збереження
        if output_dir:
            vis_image = visualize_result(image, result)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), vis_image)
        
    
    # Підсумок
    print(f"\n{'='*60}")
    print("✅ Batch обробка завершена!")
    print(f"{'='*60}")
    print(f"Оброблено:        {i} зображень")
    print(f"Positive:         {results['positive']}")
    print(f"Negative:         {results['negative']}")
    print(f"Середній час:     {(total_time/i)*1000:.2f} ms")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='ResNet18 Inference на Hailo AI HAT+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:

  # Одне зображення
  %(prog)s --model resnet18.hef --input image.jpg --show

  # Batch обробка
  %(prog)s --model resnet18.hef --batch images/ --output results/

  # Відео
  %(prog)s --model resnet18.hef --input video.mp4 --output result.mp4 --show

  # Real-time камера
  %(prog)s --model resnet18.hef --input 0 --show
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Шлях до HEF файлу')
    parser.add_argument('--batch', type=str, 
                       help='Папка для batch обробки')

    parser.add_argument('--input', type=str,
                       help='Зображення, відео або 0 для камери')
    parser.add_argument('--output', type=str,
                       help='Де зберегти результат')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    
    if not Path(args.model).exists():
        print(f"❌ HEF файл не знайдено: {args.model}")
        return
    if not args.input and not args.batch:
        parser.error("Потрібно вказати --input або --batch")
    # Ініціалізація моделі
    print(f"\n{'='*60}")
    print("🚀 ResNet18 Inference на Hailo AI HAT+")
    print(f"{'='*60}\n")
    
    model = ResNet18Inference(args.model, threshold=args.threshold)
    
    with model:
        if args.batch:
            process_batch(model, args.batch, args.output, args.show)
        else:
           
            if not Path(args.input).exists():
                print(f"❌ Файл не знайдено: {args.input}")
                return
            
            process_image(model, args.input, args.output, args.show)
    
    
    print("\n✅ Готово!")

if __name__ == '__main__':
    main()