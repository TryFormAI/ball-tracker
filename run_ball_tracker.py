import sys
import os
from pathlib import Path
import logging
import argparse
import cv2
import numpy as np
import tensorflow as tf
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the BallDetector and necessary custom objects
from ball_detector import BallDetector, detection_loss, calculate_iou_tf, calculate_ciou_tf

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balltracker/inference.log'),
        logging.StreamHandler()
    ]
)

def draw_bbox(image: np.ndarray, detections: list, target_size: tuple = None) -> np.ndarray:
    # draw bounding boxes on the image
    display_image = image.copy()
    
    # if a target_size is provided, it means the detection coordinates are for a resized image
    # and need to be scaled back to the display_image's original size
    if target_size:
        original_h, original_w = image.shape[:2]
        target_w, target_h = target_size[0], target_size[1] # target_size is (width, height)
        
        # scale factors based on the input_shape of the model (e.g., 224x224)
        scale_x = original_w / target_w
        scale_y = original_h / target_h
    else:
        # no scaling needed if detections are already in original image coordinates
        scale_x, scale_y = 1.0, 1.0

    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']

        # scale bbox coordinates if necessary
        x, y, w, h = int(bbox[0] * scale_x), int(bbox[1] * scale_y), \
                     int(bbox[2] * scale_x), int(bbox[3] * scale_y)
        
        # ensure coordinates are valid integers
        x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

        # draw rectangle
        color = (0, 255, 0) # green color for bounding box
        thickness = 2
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, thickness)

        # put text (confidence)
        text = f"Ball: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # position text slightly above the bounding box
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + h + text_size[1] + 10
        
        # draw a filled rectangle behind the text for better readability
        cv2.rectangle(display_image, (text_x, text_y - text_size[1] - 5), 
                      (text_x + text_size[0], text_y + 5), color, -1)
        cv2.putText(display_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return display_image

def process_image(detector: BallDetector, image_path: str, output_path: str = None):
    # process a single image file
    if not Path(image_path).exists():
        logging.error(f"Image file not found: {image_path}")
        return

    logging.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image: {image_path}")
        return
    
    # convert to rgb for model input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    detections = detector.detect_balls(image_rgb)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # convert to ms
    
    logging.info(f"Detection time: {inference_time:.2f} ms")
    
    # target_size is the input_shape of the model (width, height)
    display_image = draw_bbox(image, detections, target_size=(detector.input_shape[1], detector.input_shape[0]))

    if output_path:
        cv2.imwrite(output_path, display_image)
        logging.info(f"Processed image saved to: {output_path}")
    else:
        cv2.imshow("Ball Detection", display_image)
        cv2.waitKey(0) # wait indefinitely for a key press
        cv2.destroyAllWindows()

def process_video(detector: BallDetector, video_path: str, output_path: str = None, webcam_id: int = None):
    # process a video file or webcam feed
    if webcam_id is not None:
        cap = cv2.VideoCapture(webcam_id)
        logging.info(f"Opening webcam: {webcam_id}")
    else:
        if not Path(video_path).exists():
            logging.error(f"Video file not found: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        logging.info(f"Opening video file: {video_path}")

    if not cap.isOpened():
        logging.error("Error: Could not open video stream.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        logging.info(f"Writing output video to: {output_path}")

    logging.info("Starting video processing. Press 'q' to quit.")

    frame_count = 0
    total_inference_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream or error reading frame.")
            break

        frame_count += 1
        
        # convert to rgb for model input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        detections = detector.detect_balls(frame_rgb)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 # convert to ms
        total_inference_time += inference_time

        # target_size is the input_shape of the model (width, height)
        display_frame = draw_bbox(frame, detections, target_size=(detector.input_shape[1], detector.input_shape[0]))

        if output_path:
            out.write(display_frame)
        else:
            cv2.imshow("Ball Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit
                logging.info("User requested quit.")
                break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_inference_time = total_inference_time / frame_count
        logging.info(f"Average inference time per frame: {avg_inference_time:.2f} ms")
        logging.info(f"Estimated FPS (processing only): {1000 / avg_inference_time:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Run ball detection on image or video.')
    parser.add_argument('--model', type=str, default='balltracker/ball_detection_model_final.keras',
                        help='Path to the trained Keras model file.')
    parser.add_argument('--image', type=str,
                        help='Path to an input image file for detection.')
    parser.add_argument('--video', type=str,
                        help='Path to an input video file for detection.')
    parser.add_argument('--webcam', type=int,
                        help='Webcam ID (e.g., 0 for default webcam) for live detection.')
    parser.add_argument('--output', type=str,
                        help='Optional path to save the output image or video.')
    
    args = parser.parse_args()

    # ensure balltracker directory exists for logs/output
    Path('balltracker').mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("BALL TRACKER INFERENCE SCRIPT")
    logging.info("=" * 60)
    logging.info(f"Arguments: {vars(args)}")

    if not Path(args.model).exists():
        logging.error(f"Error: Model file not found at {args.model}. Please train the model first or provide a correct path.")
        sys.exit(1)

    logging.info("Loading ball detector model...")
    # custom objects needed for loading a model saved with custom loss/metrics
    detector = BallDetector()
    try:
        detector.model = tf.keras.models.load_model(
            args.model,
            custom_objects={
                'detection_loss': detection_loss,
                'calculate_iou_tf': calculate_iou_tf,
                'calculate_ciou_tf': calculate_ciou_tf, # make sure this is available if used in loss
                'MeanIoUForBBoxes': MeanIoUForBBoxes, # for custom metrics
                'ConfAccuracy': ConfAccuracy # for custom metrics
            }
        )
        logging.info(f"Model loaded successfully from {args.model}")
    except Exception as e:
        logging.error(f"Failed to load model {args.model}: {e}")
        logging.exception("Full traceback for model loading:")
        sys.exit(1)

    # warm up the model for accurate timing
    logging.info("Warming up the model...")
    dummy_image = np.zeros((detector.input_shape[0], detector.input_shape[1], detector.input_shape[2]), dtype=np.uint8)
    detector.detect_balls(dummy_image)
    logging.info("Model warm-up complete.")

    if args.image:
        process_image(detector, args.image, args.output)
    elif args.video:
        process_video(detector, args.video, args.output)
    elif args.webcam is not None:
        process_video(detector, None, args.output, args.webcam)
    else:
        logging.warning("No input specified. Use --image, --video, or --webcam.")
        parser.print_help()
    
    logging.info("=" * 60)
    logging.info("BALL TRACKER INFERENCE COMPLETE")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
