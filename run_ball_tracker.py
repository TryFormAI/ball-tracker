

import sys
import os
from pathlib import Path
import logging
import argparse
import cv2
import numpy as np
from ball_detector import BallDetector
from ball_tracker import BallTracker

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(dataset_path: str, epochs: int = 50, batch_size: int = 32):
    
    logging.info("Training ball detection model...")
    
    detector = BallDetector()
    
    annotations_file = Path(dataset_path) / "annotations.json"
    images_dir = Path(dataset_path) / "images"
    
    history = detector.train(
        coco_path=str(annotations_file),
        images_dir=str(images_dir),
        epochs=epochs,
        batch_size=batch_size
    )
    
    logging.info("Training completed!")
    return detector

def process_video(video_path: str, 
                  model_path: str = None,
                  output_path: str = None,
                  show_preview: bool = True):
    
    logging.info(f"Processing video: {video_path}")
    
    # Initialize tracker
    tracker = BallTracker(detector_model_path=model_path)
    
    # Process video
    tracker.process_video(video_path, output_path)
    
    # Get trajectory data
    trajectories = tracker.get_trajectory_data()
    
    logging.info(f"Processing complete. Found {len(trajectories)} ball trajectories.")
    
    return trajectories

def test_detection_on_image(image_path: str, model_path: str = None):
    
    logging.info(f"Testing detection on image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image: {image_path}")
        return
    
    # Initialize detector
    detector = BallDetector(model_path)
    
    # Detect balls
    detections = detector.detect_balls(image)
    
    # Draw detections
    output_image = image.copy()
    for detection in detections:
        bbox = detection['bbox']
        center = detection['center']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(output_image, 
                     (bbox[0], bbox[1]), 
                     (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                     (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(output_image, center, 5, (255, 0, 0), -1)
        
        # Draw confidence text
        cv2.putText(output_image, f"{confidence:.2f}", 
                   (bbox[0], bbox[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save output
    output_path = f"balltracker/detection_test_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, output_image)
    
    logging.info(f"Detection test complete. Output saved to: {output_path}")
    logging.info(f"Found {len(detections)} balls")
    
    return detections

def analyze_trajectory(trajectories: list):
    
    if not trajectories:
        logging.info("No trajectories to analyze")
        return
    
    print("\n=== Trajectory Analysis ===")
    
    for i, trajectory in enumerate(trajectories):
        if not trajectory:
            continue
            
        print(f"\nBall {i}:")
        print(f"  Trajectory points: {len(trajectory)}")
        
        if len(trajectory) >= 2:
            # Calculate trajectory statistics
            x_coords = [point['center'][0] for point in trajectory]
            y_coords = [point['center'][1] for point in trajectory]
            
            # Distance traveled
            total_distance = 0
            for j in range(1, len(trajectory)):
                pt1 = trajectory[j-1]['center']
                pt2 = trajectory[j]['center']
                distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                total_distance += distance
            
            print(f"  Total distance: {total_distance:.1f} pixels")
            print(f"  X range: {min(x_coords)} - {max(x_coords)}")
            print(f"  Y range: {min(y_coords)} - {max(y_coords)}")
            
            # Average confidence
            avg_confidence = np.mean([point['confidence'] for point in trajectory])
            print(f"  Average confidence: {avg_confidence:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Ball tracking system')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'process', 'test', 'analyze'],
                       help='Operation mode')
    parser.add_argument('--dataset', type=str, default='ball_dataset',
                       help='Path to ball dataset (for training)')
    parser.add_argument('--model', type=str, default='balltracker/ball_detection_model.h5',
                       help='Path to trained model')
    parser.add_argument('--video', type=str,
                       help='Path to input video (for processing)')
    parser.add_argument('--image', type=str,
                       help='Path to input image (for testing)')
    parser.add_argument('--output', type=str,
                       help='Path to output video')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable video preview')
    
    args = parser.parse_args()
    
    # Create balltracker directory if it doesn't exist
    Path('balltracker').mkdir(exist_ok=True)
    
    if args.mode == 'train':
        # Train model
        if not Path(args.dataset).exists():
            logging.error(f"Dataset not found: {args.dataset}")
            return
        
        detector = train_model(args.dataset, args.epochs, args.batch_size)
        
    elif args.mode == 'process':
        # Process video
        if not args.video:
            logging.error("Video path required for processing mode")
            return
        
        if not Path(args.video).exists():
            logging.error(f"Video not found: {args.video}")
            return
        
        trajectories = process_video(
            video_path=args.video,
            model_path=args.model if Path(args.model).exists() else None,
            output_path=args.output,
            show_preview=not args.no_preview
        )
        
        # Analyze trajectories
        analyze_trajectory(trajectories)
        
    elif args.mode == 'test':
        # Test detection on image
        if not args.image:
            logging.error("Image path required for test mode")
            return
        
        if not Path(args.image).exists():
            logging.error(f"Image not found: {args.image}")
            return
        
        detections = test_detection_on_image(args.image, args.model)
        
    elif args.mode == 'analyze':
        # Analyze dataset
        if not Path(args.dataset).exists():
            logging.error(f"Dataset not found: {args.dataset}")
            return
        
        from train_ball_detector import analyze_dataset
        analyze_dataset(args.dataset)

if __name__ == "__main__":
    main() 