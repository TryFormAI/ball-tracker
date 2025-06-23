#!/usr/bin/env python3


import sys
import os
from pathlib import Path
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from ball_detector import BallDetector
import cv2
import tensorflow as tf
import time
from datetime import datetime
from tensorflow.keras.utils import Sequence

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balltracker/training.log'),
        logging.StreamHandler()
    ]
)

class BallDatasetSequence(Sequence):
    def __init__(self, dataset_path, split, batch_size=32, input_shape=(224, 224, 3), shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.split_dir = Path(dataset_path) / split
        self.annotations_file = self.split_dir / "_annotations.coco.json"
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self._load_annotations()
        self.on_epoch_end()

    def _load_annotations(self):
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        self.img_id_to_info = {img['id']: img for img in coco_data['images']}
        self.img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        self.img_ids = list(self.img_id_to_info.keys())
        self.n = len(self.img_ids)
        logging.info(f"Loaded {self.n} images for split '{self.split_dir.name}'")

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = []
        batch_labels = []
        for i in batch_indices:
            img_id = self.img_ids[i]
            img_info = self.img_id_to_info[img_id]
            img_path = self.split_dir / img_info['file_name']
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img_normalized = img_resized / 255.0
            # Get annotations
            anns = self.img_id_to_anns.get(img_id, [])
            # Find golf ball annotation (category_id==1)
            label = np.zeros(5, dtype=np.float32)
            for ann in anns:
                if ann['category_id'] == 1:
                    bbox = ann['bbox']
                    img_h, img_w = img.shape[:2]
                    x_norm = bbox[0] / img_w
                    y_norm = bbox[1] / img_h
                    w_norm = bbox[2] / img_w
                    h_norm = bbox[3] / img_h
                    label = np.array([x_norm, y_norm, w_norm, h_norm, 1.0], dtype=np.float32)
                    break
            batch_imgs.append(img_normalized)
            batch_labels.append(label)
        # If some images failed to load, batch may be smaller
        return np.array(batch_imgs), np.array(batch_labels)

def train_model(dataset_path: str, 
                epochs: int = 50, 
                batch_size: int = 32,
                learning_rate: float = 0.001):
    logging.info("=" * 60)
    logging.info("STARTING BALL DETECTION MODEL TRAINING (MEMORY EFFICIENT)")
    logging.info("=" * 60)
    logging.info(f"Training parameters:")
    logging.info(f"  - Dataset: {dataset_path}")
    logging.info(f"  - Epochs: {epochs}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Learning rate: {learning_rate}")
    logging.info(f"  - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Data generators
    train_gen = BallDatasetSequence(dataset_path, 'train', batch_size=batch_size, shuffle=True)
    val_gen = BallDatasetSequence(dataset_path, 'valid', batch_size=batch_size, shuffle=False)

    logging.info(f"Data summary:")
    logging.info(f"  - Training batches: {len(train_gen)}")
    logging.info(f"  - Validation batches: {len(val_gen)}")
    logging.info(f"  - Input shape: {train_gen.input_shape}")

    # Initialize detector
    logging.info("Initializing ball detector...")
    detector = BallDetector()
    logging.info("Building model architecture...")
    detector.model = detector.build_model()
    logging.info("Model architecture:")
    detector.model.summary(print_fn=logging.info)
    total_params = detector.model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in detector.model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    logging.info(f"Model parameters:")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Trainable parameters: {trainable_params:,}")
    logging.info(f"  - Non-trainable parameters: {non_trainable_params:,}")
    detector.model.optimizer.learning_rate = learning_rate
    logging.info(f"Set learning rate to: {learning_rate}")

    logging.info("=" * 40)
    logging.info("STARTING TRAINING")
    logging.info("=" * 40)
    training_start_time = time.time()
    history = detector.model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint(
                'balltracker/best_model.keras',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.CSVLogger('balltracker/training_log.csv')
        ],
        verbose=1
    )
    training_time = time.time() - training_start_time
    logging.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Evaluate model
    logging.info("Evaluating model on validation set...")
    val_loss, val_acc = detector.model.evaluate(val_gen, verbose=0)
    logging.info(f"Validation results:")
    logging.info(f"  - Loss: {val_loss:.4f}")
    logging.info(f"  - Accuracy: {val_acc:.4f}")
    if history.history:
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])
        logging.info("Training history summary:")
        logging.info(f"  - Final training loss: {final_train_loss:.4f}")
        logging.info(f"  - Final training accuracy: {final_train_acc:.4f}")
        logging.info(f"  - Best validation loss: {best_val_loss:.4f}")
        logging.info(f"  - Best validation accuracy: {best_val_acc:.4f}")
        logging.info(f"  - Epochs trained: {len(history.history['loss'])}")
    logging.info("Saving trained model...")
    detector.save_model('balltracker/ball_detection_model.keras')
    logging.info("Generating training plots...")
    plot_training_history(history)
    logging.info("=" * 60)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)
    return detector

def plot_training_history(history):
    logging.info("Creating training history plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('balltracker/training_history.png', dpi=300, bbox_inches='tight')
    logging.info("Training plots saved to: balltracker/training_history.png")
    plt.show()

def evaluate_model(dataset_path: str, model_path: str, batch_size: int = 32):
    logging.info("=" * 40)
    logging.info("MODEL EVALUATION")
    logging.info("=" * 40)
    logging.info(f"Evaluating model: {model_path}")
    logging.info(f"Test dataset: {dataset_path}")
    test_gen = BallDatasetSequence(dataset_path, 'test', batch_size=batch_size, shuffle=False)
    if len(test_gen) == 0:
        logging.error("No test data found!")
        return
    logging.info(f"Test batches: {len(test_gen)}")
    logging.info("Loading trained model...")
    detector = BallDetector(model_path)
    logging.info("Running evaluation...")
    evaluation_start_time = time.time()
    test_loss, test_acc = detector.model.evaluate(test_gen, verbose=0)
    evaluation_time = time.time() - evaluation_start_time
    logging.info("Evaluation results:")
    logging.info(f"  - Test loss: {test_loss:.4f}")
    logging.info(f"  - Test accuracy: {test_acc:.4f}")
    logging.info(f"  - Evaluation time: {evaluation_time:.2f} seconds")
    logging.info("Calculating additional metrics...")
    predictions = detector.model.predict(test_gen, verbose=0)
    y_test = np.concatenate([y for _, y in test_gen], axis=0)
    bbox_mae = np.mean(np.abs(predictions[:, :4] - y_test[:, :4]))
    logging.info(f"  - Bounding box MAE: {bbox_mae:.4f}")
    conf_predictions = (predictions[:, 4] > 0.5).astype(int)
    conf_accuracy = np.mean(conf_predictions == y_test[:, 4].astype(int))
    logging.info(f"  - Confidence accuracy: {conf_accuracy:.4f}")
    logging.info("=" * 40)
    logging.info("EVALUATION COMPLETED")
    logging.info("=" * 40)
    return test_acc, test_loss

def check_dataset_structure(dataset_path: str) -> bool:
    logging.info(f"Checking dataset structure at: {dataset_path}")
    dataset_dir = Path(dataset_path)
    required_splits = ['train', 'test', 'valid']
    for split in required_splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logging.error(f"Split directory not found: {split_dir}")
            return False
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            logging.error(f"COCO annotations file not found: {annotations_file}")
            return False
        image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        if not image_files:
            logging.warning(f"No image files found in {split_dir}")
        else:
            logging.info(f"Found {len(image_files)} images in {split}")
    logging.info("Dataset structure validation passed")
    return True

def analyze_dataset(dataset_path: str):
    logging.info("Starting dataset analysis...")
    print("\n=== Dataset Analysis ===")
    for split in ['train', 'test', 'valid']:
        split_dir = Path(dataset_path) / split
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            print(f"  {split}: Not found")
            logging.warning(f"Annotations file not found for {split} split")
            continue
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        num_images = len(coco_data['images'])
        category_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"\n{split.upper()}:")
        print(f"  Total images: {num_images}")
        print(f"  Total annotations: {len(coco_data['annotations'])}")
        print("  Annotations by category:")
        for cat_id, count in category_counts.items():
            cat_name = category_names.get(cat_id, f"Unknown_{cat_id}")
            print(f"    {cat_name} (ID {cat_id}): {count}")
        image_sizes = []
        for img_info in coco_data['images']:
            image_sizes.append((img_info['width'], img_info['height']))
        if image_sizes:
            widths, heights = zip(*image_sizes)
            print(f"  Image dimensions:")
            print(f"    Min: {min(widths)}x{min(heights)}")
            print(f"    Max: {max(widths)}x{max(heights)}")
            print(f"    Average: {sum(widths)//len(widths)}x{sum(heights)//len(heights)}")
        logging.info(f"{split} split analysis: {num_images} images, {len(coco_data['annotations'])} annotations")

def main():
    parser = argparse.ArgumentParser(description='Train ball detection model')
    parser.add_argument('--dataset', type=str, default='ball_dataset',
                       help='Path to ball dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze dataset, don\'t train')
    parser.add_argument('--evaluate', type=str,
                       help='Evaluate trained model (provide model path)')
    args = parser.parse_args()
    Path('balltracker').mkdir(exist_ok=True)
    logging.info("=" * 60)
    logging.info("BALL DETECTION TRAINING SCRIPT")
    logging.info("=" * 60)
    logging.info(f"Arguments: {vars(args)}")
    if not check_dataset_structure(args.dataset):
        logging.error("Dataset validation failed. Please check the dataset structure.")
        return
    analyze_dataset(args.dataset)
    if args.analyze_only:
        logging.info("Dataset analysis complete.")
        return
    if args.evaluate:
        if not Path(args.evaluate).exists():
            logging.error(f"Model not found: {args.evaluate}")
            return
        evaluate_model(args.dataset, args.evaluate, batch_size=args.batch_size)
        return
    try:
        detector = train_model(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if detector:
            logging.info("Training completed successfully!")
            logging.info("Model saved to: balltracker/ball_detection_model.keras")
            logging.info("Evaluating on test set...")
            evaluate_model(args.dataset, 'balltracker/ball_detection_model.keras', batch_size=args.batch_size)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main() 