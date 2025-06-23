# Ball Tracking System

A comprehensive ball detection and tracking system for golf videos that can identify balls and draw tracers behind them once they've been hit.

## Features

- **Ball Detection**: CNN-based model using EfficientNetB0 for accurate ball detection
- **Ball Tracking**: Multi-object tracking with trajectory recording
- **Impact Detection**: Automatic detection of when balls are hit
- **Trajectory Visualization**: Real-time tracer drawing with color-coded trajectories
- **Video Processing**: Support for processing entire videos with output saving
- **Model Training**: Complete training pipeline for custom datasets

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the ball dataset in the correct structure:
```
ball_dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── valid/
    ├── _annotations.coco.json
    ├── image1.jpg
    └── ...
```

## Usage

### 1. Analyze Dataset

First, analyze your ball dataset to understand its structure:

```bash
python train_ball_detector.py --analyze-only --dataset ball_dataset
```

### 2. Train the Model

Train the ball detection model on your dataset:

```bash
python train_ball_detector.py --dataset ball_dataset --epochs 50 --batch-size 32
```

Training options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)

### 3. Process Videos

Process a video with ball tracking:

```bash
python run_ball_tracker.py --mode process --video path/to/video.mp4 --output balltracker/output.mp4
```

Processing options:
- `--video`: Input video path
- `--output`: Output video path (optional)
- `--model`: Path to trained model (default: balltracker/ball_detection_model.h5)
- `--no-preview`: Disable video preview

### 4. Test Detection on Images

Test ball detection on a single image:

```bash
python run_ball_tracker.py --mode test --image path/to/image.jpg
```

### 5. Evaluate Model

Evaluate a trained model on the test set:

```bash
python train_ball_detector.py --evaluate balltracker/ball_detection_model.h5 --dataset ball_dataset
```

## System Architecture

### BallDetector Class
- **Purpose**: CNN-based ball detection using EfficientNetB0
- **Input**: Images (224x224x3)
- **Output**: Bounding boxes and confidence scores
- **Features**: Custom loss function, model saving/loading

### BallTracker Class
- **Purpose**: Multi-object ball tracking with trajectory recording
- **Features**: 
  - Impact detection (significant ball movement)
  - Trajectory recording after impact
  - Real-time visualization
  - Multiple ball tracking

### Training Pipeline
- **Data Loading**: Handles COCO format with train/valid/test splits
- **Preprocessing**: Image resizing, normalization, augmentation
- **Training**: Early stopping, learning rate scheduling
- **Evaluation**: Test set evaluation with metrics

## Key Features

### Impact Detection
The system automatically detects when a ball is hit by monitoring significant movement between frames. Once impact is detected, trajectory recording begins.

### Trajectory Visualization
- Color-coded trajectories for multiple balls
- Thickness based on detection confidence
- Real-time drawing with fade effects

### Multi-Object Tracking
- Tracks multiple balls simultaneously
- Handles ball occlusion and re-detection
- Maintains ball identity across frames

### Performance Optimizations
- EfficientNetB0 backbone for speed/accuracy balance
- Batch processing for video frames
- Early stopping to prevent overfitting

## File Structure

```
balltracker/
├── ball_detector.py          # Ball detection model
├── ball_tracker.py           # Ball tracking system
├── train_ball_detector.py    # Training script
├── run_ball_tracker.py       # Main execution script
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── ball_detection_model.h5   # Trained model (after training)
└── training_history.png      # Training plots (after training)
```

## Model Architecture

The ball detection model uses:
- **Backbone**: EfficientNetB0 (pre-trained on ImageNet)
- **Head**: Custom detection head with dropout layers
- **Output**: 5 values (x, y, width, height, confidence)
- **Loss**: Combined bounding box MSE + confidence cross-entropy

## Training Data Format

The system expects COCO format annotations with:
- Images in train/test/valid folders
- `_annotations.coco.json` in each folder
- Golf ball category with ID 1
- Bounding box annotations in [x, y, width, height] format

## Troubleshooting

### Common Issues

1. **No balls detected**: 
   - Check model confidence threshold
   - Verify training data quality
   - Ensure proper image preprocessing

2. **Poor tracking performance**:
   - Adjust `max_ball_distance` parameter
   - Check video frame rate and quality
   - Verify ball detection accuracy

3. **Training issues**:
   - Check dataset structure
   - Verify COCO annotation format
   - Monitor training loss/accuracy

### Performance Tips

- Use GPU for training (TensorFlow will auto-detect)
- Reduce batch size if memory issues occur
- Adjust learning rate based on training progress
- Use data augmentation for better generalization

## Examples

### Basic Training
```bash
# Analyze dataset first
python train_ball_detector.py --analyze-only --dataset ball_dataset

# Train model
python train_ball_detector.py --dataset ball_dataset --epochs 100

# Evaluate model
python train_ball_detector.py --evaluate balltracker/ball_detection_model.h5 --dataset ball_dataset
```

### Video Processing
```bash
# Process video with preview
python run_ball_tracker.py --mode process --video golf_swing.mp4

# Process video and save output
python run_ball_tracker.py --mode process --video golf_swing.mp4 --output tracked_swing.mp4

# Process without preview (faster)
python run_ball_tracker.py --mode process --video golf_swing.mp4 --no-preview
```

### Testing
```bash
# Test on single image
python run_ball_tracker.py --mode test --image ball_image.jpg

# Test with custom model
python run_ball_tracker.py --mode test --image ball_image.jpg --model custom_model.h5
```

## Contributing

To extend the system:
1. Modify `BallDetector` for different detection architectures
2. Enhance `BallTracker` with advanced tracking algorithms
3. Add new visualization features
4. Implement additional data augmentation techniques

## License

This project is part of the golf swing analysis system. 