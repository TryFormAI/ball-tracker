import tensorflow as tf
import numpy as np
import cv2
import json
import h5py
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_iou_tf(box1, box2):
    # box: [x, y, w, h] (top-left, width, height)
    box1_x_min = box1[..., 0]
    box1_y_min = box1[..., 1]
    box1_x_max = box1[..., 0] + box1[..., 2]
    box1_y_max = box1[..., 1] + box1[..., 3]

    box2_x_min = box2[..., 0]
    box2_y_min = box2[..., 1]
    box2_x_max = box2[..., 0] + box2[..., 2]
    box2_y_max = box2[..., 1] + box2[..., 3]

    inter_x_min = tf.maximum(box1_x_min, box2_x_min)
    inter_y_min = tf.maximum(box1_y_min, box2_y_min)
    inter_x_max = tf.minimum(box1_x_max, box2_x_max)
    inter_y_max = tf.minimum(box1_y_max, box2_y_max)

    inter_width = tf.maximum(0.0, inter_x_max - inter_x_min)
    inter_height = tf.maximum(0.0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]
    union_area = box1_area + box2_area - inter_area

    # avoid division by zero
    iou = tf.where(tf.equal(union_area, 0), 0.0, inter_area / union_area)
    return iou

def detection_loss(y_true, y_pred):
    # Separate bounding box and confidence predictions
    bbox_true = y_true[:, :4]
    conf_true = y_true[:, 4]
    bbox_pred = y_pred[:, :4]
    conf_pred = y_pred[:, 4]
    
    # mask for positive samples (where an object is present in y_true)
    object_mask = tf.cast(conf_true, dtype=tf.bool)

    # bounding box loss (only for positive samples)
    bbox_loss = 0.0
    # ensure there are positive samples to avoid issues with empty tensors
    if tf.reduce_sum(tf.cast(object_mask, tf.float32)) > 0:
        bbox_true_pos = tf.boolean_mask(bbox_true, object_mask)
        bbox_pred_pos = tf.boolean_mask(bbox_pred, object_mask)

        # using 1 - IoU as localization loss for simplicity. More advanced: GIoU, DIoU, CIoU.
        iou = calculate_iou_tf(bbox_true_pos, bbox_pred_pos)
        bbox_loss = tf.reduce_mean(1.0 - iou) # mean over positive samples

    # confidence loss (Binary Crossentropy)
    # using 'from_logits=False' since sigmoid activation is used in the model output
    conf_loss = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)
    conf_loss = tf.reduce_mean(conf_loss) # mean over all samples
    
    # combined loss
    # the weight (2.0) is a hyperparameter for balancing localization and classification
    total_loss = bbox_loss + 2.0 * conf_loss
    
    return total_loss

class BallDetector:
    # Using CNN architecture for ball detection
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.input_shape = (224, 224, 3)  # Input size of the dataset - if using for other data, this may need to be changed
        self.class_names = ['background', 'golf_ball']
        self.confidence_threshold = 0.5
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else: # build model immediately if no path given, or path invalid
            logging.info("Model path not provided or model does not exist. A new model will be built.")
            self.model = self.build_model()
    
    def build_model(self) -> tf.keras.Model:
        # Use EfficientNetB0 as backbone for good performance/speed balance
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        # Unfreeze the backbone for transfer learning
        base_model.trainable = True
        
        # Add detection head
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output: bounding box (x, y, width, height) + confidence
        bbox_outputs = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(x)  # normalized coords
        conf_output = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(x)
        outputs = tf.keras.layers.Concatenate(name='detection')([bbox_outputs, conf_output])
        
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=detection_loss
        )
        logging.info("Model built successfully.")
        return model
    
    def detect_balls(self, image: np.ndarray) -> List[Dict]:
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        # Preprocess image
        img_resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        # Predict
        predictions = self.model.predict(img_batch)[0]
        # parse predictions
        x, y, w, h = predictions[:4]
        confidence = predictions[4]
        # Convert normalized co-ordinates back to image coordinates
        img_h, img_w = image.shape[:2]
        x_pixel = int(x * img_w)
        y_pixel = int(y * img_h)
        w_pixel = int(w * img_w)
        h_pixel = int(h * img_h)
        # filter by confidence
        if confidence > self.confidence_threshold:
            return [{
                'bbox': [x_pixel, y_pixel, w_pixel, h_pixel],
                'confidence': float(confidence),
                'center': (x_pixel + w_pixel//2, y_pixel + h_pixel//2)
            }]
        return []
    
    def save_model(self, model_path: str = 'balltracker/ball_detection_model.keras'):
        if self.model:
            if not model_path.endswith('.keras'):
                model_path += '.keras'
            self.model.save(model_path)
            logging.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        try:
            self.model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'detection_loss': detection_loss, 'calculate_iou_tf': calculate_iou_tf}
            )
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

if __name__ == "__main__":
    detector = BallDetector()
    # example usage (will not run without actual dataset or training)