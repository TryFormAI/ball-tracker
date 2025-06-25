import sys
import os
from pathlib import Path
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import time
from datetime import datetime
from tensorflow.keras.utils import Sequence
import albumentations as A

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

# Import the BallDetector from your module
from ball_detector import BallDetector, detection_loss, calculate_iou_tf, calculate_ciou_tf

# Define custom metrics for evaluation
# Added decorator for Keras serialization
@tf.keras.saving.register_keras_serializable()
class MeanIoUForBBoxes(tf.keras.metrics.Mean):
    def __init__(self, name='mean_iou_for_bboxes', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.num_examples = self.add_weight(name='num_examples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        bbox_true = y_true[:, :4]
        conf_true = y_true[:, 4]
        bbox_pred = y_pred[:, :4]
        # conf_pred = y_pred[:, 4] # not used in iou calculation

        # only consider positive samples (where a ball exists)
        object_mask = tf.cast(conf_true, dtype=tf.bool)
        
        # calculate sum of positive samples
        num_positive_samples = tf.reduce_sum(tf.cast(object_mask, tf.float32))

        # use tf.cond to conditionally calculate iou and update state
        # this replaces the if statement that caused the symbolic tensor error
        tf.cond(
            num_positive_samples > 0,
            lambda: self._update_positive_samples_state(bbox_true, bbox_pred, object_mask),
            lambda: None # do nothing if no positive samples
        )

    # helper method to update state for positive samples
    def _update_positive_samples_state(self, bbox_true, bbox_pred, object_mask):
        bbox_true_pos = tf.boolean_mask(bbox_true, object_mask)
        bbox_pred_pos = tf.boolean_mask(bbox_pred, object_mask)

        # calculate IoU for positive samples
        # still use calculate_iou_tf for the metric, as it's the raw overlap value
        iou = calculate_iou_tf(bbox_true_pos, bbox_pred_pos)
        
        # update total IoU and count
        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.num_examples.assign_add(tf.cast(tf.shape(iou)[0], tf.float32))


    def result(self):
        return tf.where(tf.equal(self.num_examples, 0), 0.0, self.total_iou / self.num_examples)

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.num_examples.assign(0.0)


# Added decorator for Keras serialization
@tf.keras.saving.register_keras_serializable()
class ConfAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, name='conf_accuracy', threshold=0.5, **kwargs):
        super().__init__(name=name, threshold=threshold, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        conf_true = y_true[:, 4]
        conf_pred = y_pred[:, 4]
        super().update_state(conf_true, conf_pred, sample_weight)


class BallDatasetSequence(Sequence):
    def __init__(self, dataset_path, split, batch_size=32, input_shape=(224, 224, 3), shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.split_dir = Path(dataset_path) / split
        self.annotations_file = self.split_dir / "_annotations.coco.json"
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augment = augment
        self._load_annotations()
        self.on_epoch_end()

        # define Albumentations augmentations (only if augment=True)
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                # Add more as needed, ensure bounding box safety if not using BboxParams correctly
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
            # Note: A.BboxParams expects [x_min, y_min, width, height] format for 'coco'

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
                logging.warning(f"Failed to load image: {img_path}. Skipping.")
                continue # Skip this image if loading fails
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # prepare annotations for Albumentations (if augmenting)
            # Albumentations expects [x, y, w, h] and label_fields is for category_ids
            anns = self.img_id_to_anns.get(img_id, [])
            bboxes = []
            category_ids = [] # Albumentations needs this even if dummy
            original_ball_present = False

            # assuming at most one golf ball per image for this model architecture
            target_bbox = np.zeros(4, dtype=np.float32) # [x, y, w, h] normalized
            target_confidence = 0.0 # 0.0 for no object, 1.0 for object

            for ann in anns:
                if ann['category_id'] == 1: # assuming 1 is the golf ball category ID
                    bboxes.append(ann['bbox']) # COCO format [x_top_left, y_top_left, width, height]
                    category_ids.append(ann['category_id']) # Dummy, as we only care about golf ball
                    original_ball_present = True
                    break # IMPORTANT: ONLY TAKE THE FIRST GOLF BALL IF MULTIPLE ARE PRESENT

            if self.augment and original_ball_present:
                # apply augmentations with bounding box awareness
                # the transform expects original image size and bbox in coco format [x,y,w,h]
                augmented = self.transform(image=img, bboxes=bboxes, category_ids=category_ids)
                img = augmented['image']
                # after augmentation, bboxes are updated. If multiple bboxes were given,
                # albumentations returns multiple. Since we only pass one, we expect one back.
                if augmented['bboxes']:
                    # take the first (and only) augmented bbox
                    bbox_aug = augmented['bboxes'][0]
                    img_h_aug, img_w_aug = img.shape[:2] # get dimensions after augmentation
                    
                    # normalize the augmented bbox relative to the augmented image size
                    target_bbox[0] = bbox_aug[0] / img_w_aug # x_norm
                    target_bbox[1] = bbox_aug[1] / img_h_aug # y_norm
                    target_bbox[2] = bbox_aug[2] / img_w_aug # w_norm
                    target_bbox[3] = bbox_aug[3] / img_h_aug # h_norm
                    target_confidence = 1.0
                else:
                    # if augmentation removed the bbox (e.g., cropped it out), treat as no object
                    target_confidence = 0.0
            elif original_ball_present:
                # no augmentation, just normalize original bbox
                bbox = bboxes[0] # Take the first (and only) bbox
                img_h, img_w = img.shape[:2]
                target_bbox[0] = bbox[0] / img_w
                target_bbox[1] = bbox[1] / img_h
                target_bbox[2] = bbox[2] / img_w
                target_bbox[3] = bbox[3] / img_h
                target_confidence = 1.0
            
            # resize image AFTER augmentation (or if no augmentation)
            img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img_normalized = img_resized / 255.0

            # the final label is [x_norm, y_norm, w_norm, h_norm, confidence]
            label = np.concatenate((target_bbox, [target_confidence]), axis=0).astype(np.float32)

            batch_imgs.append(img_normalized)
            batch_labels.append(label)

        # if some images failed to load, make sure batches are consistent
        if not batch_imgs: # Handle case where all images in batch failed
            return np.array([]), np.array([]) 

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
    train_gen = BallDatasetSequence(dataset_path, 'train', batch_size=batch_size, shuffle=True, augment=True)
    val_gen = BallDatasetSequence(dataset_path, 'valid', batch_size=batch_size, shuffle=False, augment=False)

    logging.info(f"Data summary:")
    logging.info(f"  - Training batches: {len(train_gen)}")
    logging.info(f"  - Validation batches: {len(val_gen)}")
    logging.info(f"  - Input shape: {train_gen.input_shape}")

    # Initialize detector
    logging.info("Initializing ball detector...")
    detector = BallDetector()
    logging.info("Building model architecture...")
    detector.model = detector.build_model()
    
    # Recompile with custom metrics (or include them in build_model)
    detector.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=detection_loss,
        metrics=[MeanIoUForBBoxes(), ConfAccuracy()]
    )

    logging.info("Model architecture:")
    detector.model.summary(print_fn=logging.info)
    total_params = detector.model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in detector.model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    logging.info(f"Model parameters:")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Trainable parameters: {trainable_params:,}")
    logging.info(f"  - Non-trainable parameters: {non_trainable_params:,}")
    
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
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss', min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint(
                'balltracker/best_model.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger('balltracker/training_log.csv')
        ],
        verbose=1
    )
    training_time = time.time() - training_start_time
    logging.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Evaluate model (using the best saved model)
    logging.info("Loading best model for final evaluation and saving...")
    detector.load_model('balltracker/best_model.keras')

    logging.info("Evaluating model on validation set (using best weights)...")
    # Evaluate with the metrics defined in compile
    val_results = detector.model.evaluate(val_gen, verbose=0)
    val_metrics_dict = dict(zip(detector.model.metrics_names, val_results))

    logging.info("Validation results (from best model):")
    for name, value in val_metrics_dict.items():
        logging.info(f"  - {name}: {value:.4f}")

    if history.history:
        logging.info("Training history summary:")
        logging.info(f"  - Final training loss: {history.history['loss'][-1]:.4f}")
        # the accuracy metric names change with custom metrics
        if 'conf_accuracy' in history.history:
            logging.info(f"  - Final training confidence accuracy: {history.history['conf_accuracy'][-1]:.4f}")
        if 'mean_iou_for_bboxes' in history.history:
            logging.info(f"  - Final training BBox Mean IoU: {history.history['mean_iou_for_bboxes'][-1]:.4f}")

        if 'val_loss' in history.history:
            best_val_loss = min(history.history['val_loss'])
            logging.info(f"  - Best validation loss (from history): {best_val_loss:.4f}")
            if 'val_conf_accuracy' in history.history:
                best_val_conf_acc = max(history.history['val_conf_accuracy'])
                logging.info(f"  - Best validation confidence accuracy (from history): {best_val_conf_acc:.4f}")
            if 'val_mean_iou_for_bboxes' in history.history:
                best_val_iou = max(history.history['val_mean_iou_for_bboxes'])
                logging.info(f"  - Best validation BBox Mean IoU (from history): {best_val_iou:.4f}")

        logging.info(f"  - Epochs trained: {len(history.history['loss'])}")
    
    # save the best model with a definitive name
    detector.save_model('balltracker/ball_detection_model_final.keras')
    logging.info("Generating training plots...")
    plot_training_history(history) # plot using history from fit, not loaded model
    logging.info("=" * 60)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)
    return detector

def plot_training_history(history):
    logging.info("Creating training history plots...")
    
    # dynamically find the loss and metric names
    loss_name = 'loss'
    val_loss_name = 'val_loss'
    
    # look for 'mean_iou_for_bboxes' and 'conf_accuracy'
    metric_names = [name for name in history.history.keys() if name not in [loss_name, val_loss_name] and not name.startswith('val_')]
    val_metric_names = [name for name in history.history.keys() if name.startswith('val_') and name != val_loss_name]

    if not metric_names:
        logging.warning("No custom metrics found in history to plot besides loss.")
        # if no other metrics, just plot loss
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.plot(history.history[loss_name], label='Training Loss', linewidth=2)
        if val_loss_name in history.history:
            ax1.plot(history.history[val_loss_name], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('balltracker/training_history_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        return

    # if metrics are available, plot both loss and metrics
    fig, axes = plt.subplots(1, 1 + len(metric_names), figsize=(6 * (1 + len(metric_names)), 4))
    if len(metric_names) == 1: # If only one metric, axes is not an array for the metric plot
        axes = [axes] # Make it an array for consistent indexing

    # plot Loss
    ax_loss = axes[0]
    ax_loss.plot(history.history[loss_name], label='Training Loss', linewidth=2)
    if val_loss_name in history.history:
        ax_loss.plot(history.history[val_loss_name], label='Validation Loss', linewidth=2)
    ax_loss.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    # plot other metrics
    for i, metric_name in enumerate(metric_names):
        ax_metric = axes[i + 1]
        val_metric_name = f'val_{metric_name}'
        
        ax_metric.plot(history.history[metric_name], label=f'Training {metric_name.replace("_", " ").title()}', linewidth=2)
        if val_metric_name in history.history:
            ax_metric.plot(history.history[val_metric_name], label=f'Validation {metric_name.replace("_", " ").title()}', linewidth=2)
        
        ax_metric.set_title(f'Model {metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax_metric.set_xlabel('Epoch', fontsize=12)
        ax_metric.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax_metric.legend(fontsize=10)
        ax_metric.grid(True, alpha=0.3)

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
    
    # no augmentation for evaluation
    test_gen = BallDatasetSequence(dataset_path, 'test', batch_size=batch_size, shuffle=False, augment=False)
    
    if len(test_gen) == 0:
        logging.error("No test data found!")
        return
    logging.info(f"Test batches: {len(test_gen)}")
    
    logging.info("Loading trained model...")
    # pass custom objects for metrics as well during loading
    detector = BallDetector()
    try:
        detector.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'detection_loss': detection_loss,
                'calculate_iou_tf': calculate_iou_tf, # make sure this is passed if used by loss/metrics
                'MeanIoUForBBoxes': MeanIoUForBBoxes, # for loading the custom metric
                'ConfAccuracy': ConfAccuracy, # for loading the custom metric
                'calculate_ciou_tf': calculate_ciou_tf # new custom object
            }
        )
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        logging.exception("Full traceback for model loading:")
        return

    logging.info("Running evaluation...")
    evaluation_start_time = time.time()
    # evaluate using the same metrics as during training
    test_results = detector.model.evaluate(test_gen, verbose=0)
    evaluation_time = time.time() - evaluation_start_time
    
    test_metrics_dict = dict(zip(detector.model.metrics_names, test_results))

    logging.info("Evaluation results:")
    for name, value in test_metrics_dict.items():
        logging.info(f"  - {name}: {value:.4f}")
    
    logging.info(f"  - Evaluation time: {evaluation_time:.2f} seconds")
    
    # you can add more detailed, post-evaluation metrics here if needed,
    # such as calculating precise precision/recall/F1 based on confidence and IoU thresholds.
    # however, the custom metrics `MeanIoUForBBoxes` and `ConfAccuracy` already give good indicators.

    logging.info("=" * 40)
    logging.info("EVALUATION COMPLETED")
    logging.info("=" * 40)
    
    # return values for consistency, though dictionary is more informative
    return test_metrics_dict.get('loss'), test_metrics_dict.get('conf_accuracy'), test_metrics_dict.get('mean_iou_for_bboxes')

def check_dataset_structure(dataset_path: str) -> bool:
    logging.info(f"Checking dataset structure at: {dataset_path}")
    dataset_dir = Path(dataset_path)
    required_splits = ['train', 'test', 'valid']
    all_ok = True
    for split in required_splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logging.error(f"Split directory not found: {split_dir}")
            all_ok = False
            continue
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            logging.error(f"COCO annotations file not found: {annotations_file}")
            all_ok = False
            continue
        
        # check if annotations file is empty or malformed
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            if not isinstance(coco_data, dict) or 'images' not in coco_data or 'annotations' not in coco_data:
                logging.error(f"Annotations file {annotations_file} is not a valid COCO JSON.")
                all_ok = False
                continue
            if not coco_data['images']:
                logging.warning(f"No images listed in {annotations_file}.")
            if not coco_data['annotations']:
                logging.warning(f"No annotations listed in {annotations_file}.")

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {annotations_file}: {e}")
            all_ok = False
            continue

        image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        if not image_files:
            logging.warning(f"No image files (*.jpg, *.png) found in {split_dir}")
        else:
            logging.info(f"Found {len(image_files)} images in {split_dir.name}")
    
    if all_ok:
        logging.info("Dataset structure validation passed.")
    else:
        logging.error("Dataset structure validation failed.")
    return all_ok

def analyze_dataset(dataset_path: str):
    logging.info("Starting dataset analysis...")
    print("\n=== Dataset Analysis ===")
    for split in ['train', 'test', 'valid']:
        split_dir = Path(dataset_path) / split
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            print(f"  {split}: Annotations file not found.")
            logging.warning(f"Annotations file not found for {split} split")
            continue
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  {split}: Error reading annotations: {e}")
            logging.error(f"Error decoding JSON from {annotations_file}: {e}")
            continue

        num_images = len(coco_data.get('images', []))
        category_counts = {}
        bbox_widths = []
        bbox_heights = []

        for ann in coco_data.get('annotations', []):
            cat_id = ann.get('category_id')
            if cat_id is not None:
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            if 'bbox' in ann and len(ann['bbox']) == 4:
                bbox_widths.append(ann['bbox'][2])
                bbox_heights.append(ann['bbox'][3])

        category_names = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        print(f"\n{split.upper()}:")
        print(f"  Total images: {num_images}")
        print(f"  Total annotations: {len(coco_data.get('annotations', []))}")
        print("  Annotations by category:")
        if category_counts:
            for cat_id, count in category_counts.items():
                cat_name = category_names.get(cat_id, f"Unknown_{cat_id}")
                print(f"    {cat_name} (ID {cat_id}): {count}")
        else:
            print("    No categories found.")
        
        image_sizes = []
        for img_info in coco_data.get('images', []):
            if 'width' in img_info and 'height' in img_info:
                image_sizes.append((img_info['width'], img_info['height']))
        
        if image_sizes:
            widths, heights = zip(*image_sizes)
            print(f"  Image dimensions:")
            print(f"    Min: {min(widths)}x{min(heights)}")
            print(f"    Max: {max(widths)}x{max(heights)}")
            print(f"    Average: {sum(widths)//len(widths)}x{sum(heights)//len(heights)}")
        else:
            print("  No image dimensions found.")

        if bbox_widths and bbox_heights:
            print(f"  Bounding Box dimensions (original image scale):")
            print(f"    Width - Min: {min(bbox_widths):.2f}, Max: {max(bbox_widths):.2f}, Avg: {np.mean(bbox_widths):.2f}")
            print(f"    Height - Min: {min(bbox_heights):.2f}, Max: {max(bbox_heights):.2f}, Avg: {np.mean(bbox_heights):.2f}")
            print(f"    Aspect Ratio (W/H) - Min: {(np.min(np.array(bbox_widths)/np.array(bbox_heights))):.2f}, Max: {(np.max(np.array(bbox_widths)/np.array(bbox_heights))):.2f}, Avg: {(np.mean(np.array(bbox_widths)/np.array(bbox_heights))):.2f}")
        else:
            print("  No bounding box data found for analysis.")

        logging.info(f"{split} split analysis: {num_images} images, {len(coco_data.get('annotations', []))} annotations")
    logging.info("Dataset analysis completed.")


def main():
    parser = argparse.ArgumentParser(description='Train ball detection model')
    parser.add_argument('--dataset', type=str, default='ball_dataset',
                        help='Path to ball dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze dataset, don\'t train')
    parser.add_argument('--evaluate', type=str,
                        help='Evaluate trained model (provide model path)')
    args = parser.parse_args()

    Path('balltracker').mkdir(exist_ok=True) # ensure the output directory exists

    logging.info("=" * 60)
    logging.info("BALL DETECTION TRAINING SCRIPT")
    logging.info("=" * 60)
    logging.info(f"Arguments: {vars(args)}")

    if not check_dataset_structure(args.dataset):
        logging.error(f"Dataset validation failed. Please check the dataset structure at {args.dataset}.")
        sys.exit(1) # exit if dataset structure is bad

    analyze_dataset(args.dataset)

    if args.analyze_only:
        logging.info("Dataset analysis complete. Exiting as --analyze-only was specified.")
        return

    if args.evaluate:
        if not Path(args.evaluate).exists():
            logging.error(f"Model not found for evaluation: {args.evaluate}")
            sys.exit(1)
        evaluate_model(args.dataset, args.evaluate, batch_size=args.batch_size)
        return

    try:
        # train_model will save the best model to 'balltracker/ball_detection_model_final.keras'
        detector = train_model(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if detector:
            logging.info("Training process completed.")
            logging.info("Proceeding to final evaluation on the test set.")
            # evaluate the best saved model after training
            evaluate_model(args.dataset, 'balltracker/ball_detection_model_final.keras', batch_size=args.batch_size)
    except Exception as e:
        logging.error(f"An unhandled error occurred during training: {e}")
        logging.exception("Full traceback:")
        sys.exit(1) # exit with error code

if __name__ == "__main__":
    main()
