import sys
import os
from pathlib import Path
import logging
import argparse

# Add YOLOv11 import
from yolov11 import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Train ball detection model with YOLOv11')
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
    args = parser.parse_args()

    Path('models').mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("YOLOv11 BALL DETECTION TRAINING SCRIPT")
    logging.info("=" * 60)
    logging.info(f"Arguments: {vars(args)}")

    # Setup YOLOv11 model
    model = YOLO('yolov11.yaml')  # or use a custom config if needed
    if args.weights:
        model.load(args.weights)

    # Train
    results = model.train(
        data={
            'train': str(Path(args.dataset) / 'train'),
            'val': str(Path(args.dataset) / 'valid'),
            'test': str(Path(args.dataset) / 'test'),
            'nc': 1,  # number of classes
            'names': ['golf_ball']
        },
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        lr0=args.learning_rate,
        project='models',
        name='yolov11_ball',
        exist_ok=True
    )

    logging.info("Training complete. Best weights and results saved in the 'models/yolov11_ball' directory.")

if __name__ == "__main__":
    main()
