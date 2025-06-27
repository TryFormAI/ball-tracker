import sys
import os
from pathlib import Path
import logging
import argparse

# Add YOLO import (corrected)
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Train ball detection model with YOLO')
    parser.add_argument('--dataset', type=str, default='data',
                        help='Path to ball dataset directory (should contain train, valid, test folders in YOLO format)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weights', type=str, default='',
                        help='Path to pretrained weights (optional)')
    parser.add_argument('--export-format', type=str, default='tflite',
                        choices=['tflite', 'onnx', 'torchscript', 'coreml'],
                        help='Export format for mobile/edge deployment (default: tflite)')
    args = parser.parse_args()

    Path('models').mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("YOLO BALL DETECTION TRAINING SCRIPT")
    logging.info("=" * 60)
    logging.info(f"Arguments: {vars(args)}")

    # Setup YOLO model
    if args.weights:
        model = YOLO(args.weights)  # Load from provided weights
        logging.info(f"Loaded YOLO model from weights: {args.weights}")
    else:
        model = YOLO("yolo11n.pt")
        logging.info("Initialized new YOLO model (yolo11n.pt)")

    # Train
    data_yaml_path = str(Path(args.dataset) / 'data.yaml')
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        lr0=args.learning_rate,
        project='models',
        name='yolo_ball',
        exist_ok=True
    )

    logging.info("Training complete. Best weights and results saved in the 'models/yolo_ball' directory.")

    # Export the model for mobile/edge deployment
    export_path = f"models/yolo_ball/best_model.{args.export_format}"
    logging.info(f"Exporting model to {args.export_format} format for mobile/edge deployment...")
    model.export(format=args.export_format, dynamic=False, optimize=True, half=False, int8=False, imgsz=640, device='cpu',
                 project='models', name='yolo_ball', exist_ok=True)
    logging.info(f"Model exported to {args.export_format} format. See 'models/yolo_ball' directory.")

if __name__ == "__main__":
    main()
