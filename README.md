# Golf Ball and Club Detection System

A comprehensive computer vision system for detecting and tracking golf balls and clubs in videos using ONNX-based YOLO models. This system can identify golf balls and segment golf clubs with precise outlines in real-time video processing.

## Features

- **Golf Ball Detection**: ONNX-based YOLO model for accurate ball detection and tracking
- **Golf Club Segmentation**: Advanced segmentation model that draws precise club outlines
- **Real-time Processing**: Fast inference using ONNX Runtime for live video feeds
- **Multi-format Support**: Process videos, images, or webcam feeds
- **Customizable Output**: Configurable confidence thresholds and visualization options
- **Pre-trained Models**: Includes trained models for both ball detection and club segmentation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required models:
```
models/
├── best_ball.onnx      # Pre-trained golf ball detection model
└── best_club.onnx      # Pre-trained golf club segmentation model
```

## Usage

### Golf Ball Detection

Process videos or images for ball detection:

```bash
# Process a video file
python run_ball_tracker.py --video path/to/video.mp4 --output output_video.mp4

# Process a single image
python run_ball_tracker.py --image path/to/image.jpg --output output_image.jpg

# Use webcam for live detection
python run_ball_tracker.py --webcam 0

# Use custom model
python run_ball_tracker.py --model models/best_ball.onnx --video golf_swing.mp4
```

### Golf Club Segmentation

Segment and draw club outlines:

```bash
# Process video with club segmentation
python club_detect.py --model models/best_club.onnx --video golf_swing.mp4 --output club_output.mp4

# Process single image
python club_detect.py --model models/best_club.onnx --image golf_image.jpg --output club_image.jpg

# Use webcam for live club detection
python club_detect.py --model models/best_club.onnx --webcam 0

# Customize confidence threshold and outline appearance
python club_detect.py --model models/best_club.onnx --video golf_swing.mp4 --conf 0.3 --outline-color "(255,0,0)" --outline-thickness 3
```

## Command Line Options

### Ball Detection (`run_ball_tracker.py`)
- `--model`: Path to ONNX model file (default: `balltracker/ball_detection_model_final.keras`)
- `--video`: Input video file path
- `--image`: Input image file path
- `--webcam`: Webcam ID (e.g., 0 for default webcam)
- `--output`: Output file path for processed video/image

### Club Segmentation (`club_detect.py`)
- `--model`: Path to ONNX segmentation model file (required)
- `--video`: Input video file path
- `--image`: Input image file path
- `--webcam`: Webcam ID for live processing
- `--output`: Output file path
- `--conf`: Confidence threshold (0.0 to 1.0, default: 0.5)
- `--outline-color`: BGR color for outlines (default: "(0,255,0)" for green)
- `--outline-thickness`: Thickness of outline (default: 2)

## System Architecture

### BallDetector Class (`ball_detector.py`)
- **Purpose**: ONNX-based YOLO model for golf ball detection
- **Input**: Images (640x640x3 RGB)
- **Output**: Bounding boxes with confidence scores
- **Features**: 
  - Non-Maximum Suppression (NMS) for duplicate removal
  - Confidence thresholding
  - Coordinate transformation utilities

### ClubSegmenter Class (`club_detect.py`)
- **Purpose**: ONNX-based YOLO segmentation model for golf club detection
- **Input**: Images (640x640x3 RGB)
- **Output**: Precise club outlines and bounding boxes
- **Features**:
  - Mask generation from segmentation coefficients
  - Contour extraction for smooth outlines
  - Real-time processing capabilities

### Processing Pipeline
1. **Image Preprocessing**: Resize to 640x640, normalize to 0-1 range
2. **ONNX Inference**: Run model inference using ONNX Runtime
3. **Post-processing**: Apply NMS, confidence filtering, coordinate scaling
4. **Visualization**: Draw bounding boxes or segmentation outlines
5. **Output**: Save processed video/image or display live feed

## Model Details

### Ball Detection Model
- **Architecture**: YOLO-based detection model
- **Input Size**: 640x640 pixels
- **Classes**: Golf ball (single class)
- **Output**: Bounding boxes with confidence scores

### Club Segmentation Model
- **Architecture**: YOLO-based segmentation model
- **Input Size**: 640x640 pixels
- **Classes**: Golf club (single class)
- **Output**: Segmentation masks and bounding boxes

## Performance

- **Real-time Processing**: Optimized for live video feeds
- **ONNX Runtime**: Fast inference using optimized execution providers
- **Memory Efficient**: Minimal memory footprint for embedded applications
- **Cross-platform**: Works on Windows, macOS, and Linux

## File Structure

```
ball-tracker/
├── ball_detector.py          # Ball detection implementation
├── club_detect.py            # Club segmentation implementation
├── run_ball_tracker.py       # Ball detection execution script
├── train_ball_detector.py    # Training utilities (for reference)
├── models/
│   ├── best_ball.onnx        # Pre-trained ball detection model
│   └── best_club.onnx        # Pre-trained club segmentation model
├── videos/                   # Sample videos for testing
├── outputs/                  # Output directory for processed files
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Examples

### Basic Ball Detection
```bash
# Detect balls in a golf video
python run_ball_tracker.py --video golf_swing.mp4 --output ball_tracked.mp4

# Live ball detection from webcam
python run_ball_tracker.py --webcam 0
```

### Advanced Club Segmentation
```bash
# Segment clubs with custom settings
python club_detect.py --model models/best_club.onnx --video golf_swing.mp4 \
    --conf 0.4 --outline-color "(0,0,255)" --outline-thickness 3

# Process single image for club detection
python club_detect.py --model models/best_club.onnx --image golf_image.jpg
```

### Combined Workflow
```bash
# First detect balls
python run_ball_tracker.py --video input.mp4 --output ball_output.mp4

# Then segment clubs
python club_detect.py --model models/best_club.onnx --video ball_output.mp4 --output final_output.mp4
```

## Troubleshooting

### Common Issues
1. **Model not found**: Ensure ONNX model files are in the correct location
2. **CUDA errors**: Use CPU execution provider if GPU is not available
3. **Video codec issues**: Try different output formats (mp4, avi)
4. **Performance issues**: Lower confidence thresholds or reduce input resolution

### Performance Tips
- Use `--conf` to adjust detection sensitivity
- Process videos without preview for faster execution
- Consider using GPU execution providers for better performance
- Batch process multiple videos for efficiency

## License

This project is licensed under the MIT License - see the LICENSE file for details.