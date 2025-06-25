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

def detection_loss(y_true, y_pred):
    # Separate bounding box and confidence predictions
    bbox_true = y_true[:, :4]
    conf_true = y_true[:, 4]
    bbox_pred = y_pred[:, :4]
    conf_pred = y_pred[:, 4]
    
    # Bounding box loss (MSE)
    mse = tf.keras.losses.MeanSquaredError()
    bbox_loss = mse(bbox_true, bbox_pred)
    
    # Confidence loss (Binary crossentropy)
    conf_loss = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)
    
    # Combined loss
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
    
    def build_model(self) -> tf.keras.Model:
        # Use EfficientNetB0 as backbone for good performance/speed balance
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=detection_loss,
            metrics=['accuracy']
        )
        
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
        print("Model raw predictions:", predictions)  # Debug print
        # Parse predictions
        x, y, w, h = predictions[:4]
        confidence = predictions[4]
        # Convert normalized co-ordinates back to image coordinates
        img_h, img_w = image.shape[:2]
        x_pixel = int(x * img_w)
        y_pixel = int(y * img_h)
        w_pixel = int(w * img_w)
        h_pixel = int(h * img_h)
        print(f"Bounding box (pixels): x={x_pixel}, y={y_pixel}, w={w_pixel}, h={h_pixel}, confidence={confidence}")  # Debug print
        # Filter by confidence
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
                custom_objects={'detection_loss': detection_loss}
            )
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

if __name__ == "__main__":
    detector = BallDetector()
    
    # Load pre-trained model
    # detector.load_model('balltracker/ball_detection_model.h5') 