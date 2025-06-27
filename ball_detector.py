import numpy as np
import cv2
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
import onnxruntime as ort
from collections import deque

# Configure logging for detailed debugging.
# Set level to logging.INFO for general progress messages.
# Uncomment the line below to activate very detailed frame-by-frame and detection logs.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.DEBUG) # Keeping debug enabled as requested for tracing

# --- Helper functions for ONNX model output post-processing ---
# YOLO models typically output raw predictions that need to be processed
# to get final, clean bounding boxes. This involves filtering by confidence,
# converting coordinate formats, and applying Non-Maximum Suppression (NMS).

def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(), # Not used in this implementation, part of original Ultralytics signature
    max_det: int = 300,
    nm: int = 0,  # Number of masks (0 for detection models)
) -> List[np.ndarray]:
    """
    Performs Non-Maximum Suppression (NMS) on a set of bounding boxes.
    This function filters out overlapping boxes, keeping only the best ones.

    Args:
        prediction (np.ndarray): The raw model output. For our single-class YOLOv11 ONNX,
                                 this is typically (1, num_predictions, 7) after initial reshape,
                                 where 7 is (x, y, w, h, obj_conf, class_prob, class_id).
        conf_thres (float): Confidence threshold. Detections below this are discarded.
        iou_thres (float): IoU threshold for NMS. Overlapping boxes with IoU higher than this are suppressed.
        classes (list[int]): Optional list of class IDs to consider.
        # Other arguments are mostly for compatibility with the original Ultralytics NMS function.

    Returns:
        List[np.ndarray]: A list of detections, where each element is an array (n, 6).
                          'n' is the number of detected objects for that image,
                          and each detection is (x1, y1, x2, y2, confidence, class_id).
    """
    
    # Filter predictions based on objectness confidence
    # prediction[..., 4] contains the objectness confidence score
    xc = prediction[..., 4] > conf_thres  # Get candidate boxes with sufficient confidence

    # NMS settings, mostly defaults from Ultralytics
    max_nms = 30000  # Maximum number of boxes to feed into torchvision.ops.nms()

    output = [np.zeros((0, 6), dtype=np.float32)] * prediction.shape[0] # Initialize empty list for results

    for xi, x in enumerate(prediction):  # Iterate through each image in the batch (usually just one)
        x = x[xc[xi]]  # Apply the initial confidence thresholding

        # If no boxes remain after thresholding, skip to the next image
        if not x.shape[0]:
            continue

        # Convert box coordinates from (center x, center y, width, height) to (x1, y1, x2, y2)
        # This is typically done before NMS.
        logging.debug(f"Before xywh2xyxy - raw box example: {x[0, :4]}") # Debug raw box coords before conversion
        box = xywh2xyxy(x[:, :4]) # Apply conversion
        logging.debug(f"After xywh2xyxy - converted box example: {box[0, :]}") # Debug converted box coords

        # Combine bounding box coordinates (x1,y1,x2,y2), confidence, and class ID
        # x[:, 5] is the class confidence (which is obj_conf for single class)
        # x[:, 6] is the class ID (which is 0 for our single class)
        conf_val = x[:, 5]  # Get the confidence score for NMS
        class_idx = x[:, 6] # Get the class ID
        
        # Concatenate: [x1, y1, x2, y2, confidence, class_id]
        x = np.concatenate((box, conf_val[:, np.newaxis], class_idx[:, np.newaxis]), axis=1)

        # Filter by specific class IDs if provided (e.g., classes=[0] for 'golf_ball')
        if classes is not None:
            # Keep only boxes whose class ID is in the 'classes' list
            x = x[(np.isin(x[:, 5], classes))] # Assuming class_id is at index 5 in this combined array (0-indexed)

        n = x.shape[0]  # Number of boxes remaining after class filtering
        if not n:
            continue

        # Sort detections by confidence score in descending order
        x = x[np.argsort(-x[:, 4])] # Sort by the confidence score (index 4)
        if n > max_nms:
            x = x[:max_nms]  # Limit the number of boxes for NMS processing

        # --- Manual NMS implementation (simple greedy NMS) ---
        keep = []  # List to store indices of boxes to keep
        indexes = np.arange(n) # Array of original indices

        while indexes.size > 0:
            i = indexes[0]  # Select the box with the highest confidence
            keep.append(i)  # Add its original index to our 'keep' list

            if indexes.size == 1:  # If only one box is left, we're done
                break
            
            # Calculate IoU between the current best box and all other remaining boxes
            ious = bbox_iou(x[i, :4], x[indexes[1:], :4])

            # Find indices of boxes that have high overlap (IoU > threshold)
            # These are the boxes we want to suppress.
            remove_indexes = np.where(ious > iou_thres)[0] + 1 # +1 because indexes[1:] is a sub-array

            # Remove the suppressed boxes and the current box from consideration for future iterations
            indexes = np.delete(indexes, remove_indexes) # Remove highly overlapping boxes
            indexes = np.delete(indexes, 0) # Remove the current box itself

        output[xi] = x[keep] # Store the final, non-overlapping detections for this image

    return output


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from (center_x, center_y, width, height)
    to (x1, y1, x2, y2) where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right.
    """
    y = np.copy(x) # Create a copy to avoid modifying the original array
    
    # Debug input values for transformation
    logging.debug(f"xywh2xyxy input: {x[0] if x.shape[0] > 0 else 'empty array'}") 

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # Calculate x1: center_x - width / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # Calculate y1: center_y - height / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # Calculate x2: center_x + width / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # Calculate y2: center_y + height / 2

    # Debug output values after transformation
    logging.debug(f"xywh2xyxy output: {y[0] if y.shape[0] > 0 else 'empty array'}")
    return y


def bbox_iou(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculates Intersection over Union (IoU) between a single bounding box (box1)
    and an array of other bounding boxes (boxes2).
    """
    # Extract coordinates for box1
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    
    # Calculate area of box1
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    # Extract coordinates for all boxes in boxes2
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Determine the coordinates of the intersection rectangle
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    # Calculate the intersection area (ensure width/height are non-negative)
    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate the union area
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    # Compute IoU (add a tiny epsilon to prevent division by zero for perfect overlaps)
    iou = inter_area / (union_area + 1e-6)
    return iou


# --- The Ball Detector Class ---
class BallDetector:
    """
    A class that wraps an ONNX YOLO model to detect balls.
    """
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initializes the detector with the ONNX model path and confidence threshold.

        Args:
            model_path (str): Path to the trained ONNX model file (e.g., 'best.onnx').
            confidence_threshold (float): Minimum confidence for a detection to be considered valid.
        """
        # Basic check to make sure it's an ONNX file
        if not model_path.endswith('.onnx'):
            raise ValueError("model_path must point to an ONNX file (ending with .onnx).")
        
        self.ort_session: Optional[ort.InferenceSession] = None 
        self.confidence_threshold = confidence_threshold
        
        # IMPORTANT: These class names must exactly match the 'names' list in your data.yaml
        # that was used during your model's training. The order is critical!
        # If your data.yaml had: names: ['ball'], then use ['ball'].
        self.class_names = ['golf_ball'] # Keeping this as per your earlier scripts.

        self.load_model(model_path) # Load the model when the detector is created
        logging.info(f"BallDetector initialized with ONNX model: '{model_path}' and confidence threshold: {self.confidence_threshold}")

    def load_model(self, model_path: str):
        """
        Loads the ONNX model using ONNX Runtime.
        Will raise an error if the model file is not found or cannot be loaded.
        """
        # First, check if the model file actually exists
        if not Path(model_path).is_file():
            logging.critical(f"Error: Model file not found at: {model_path}. Please verify the path.")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Create an ONNX Runtime inference session.
            # 'CPUExecutionProvider' is chosen for broad compatibility. You could use 'CUDAExecutionProvider'
            # if running on a compatible GPU setup and ONNX Runtime was built with CUDA support.
            self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            logging.info(f"ONNX model successfully loaded from {model_path}.")
        except Exception as e:
            logging.critical(f"Failed to load ONNX model from {model_path}. Ensure it's a valid ONNX file. Error: {e}")
            self.ort_session = None # Clear the session to indicate failure
            raise # Re-raise the exception to stop execution if model loading fails

    def detect_balls(self, image: np.ndarray) -> List[Dict]:
        """
        Detects balls in the given image frame using the loaded ONNX model.

        Args:
            image (np.ndarray): The input image frame (expected to be OpenCV BGR format).

        Returns:
            List[Dict]: A list of dictionaries, each describing a detected ball.
                        Includes 'bbox' (x, y, w, h), 'confidence', 'center', 'class_id', and 'class_name'.
        """
        detections = []
        if self.ort_session is None:
            logging.critical("Error: ONNX model is not loaded. Cannot perform detection.")
            raise ValueError("ONNX model not loaded. Check model loading errors logged previously.")

        # --- Image Preprocessing for ONNX Input ---
        # The ONNX model expects a specific input format (e.g., RGB, resized, normalized, CHW, with batch dim).
        
        # Convert BGR (OpenCV default) to RGB (YOLO/ONNX common expectation)
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.debug("Input image is 3-channel BGR. Converted to RGB for ONNX model input.")
        elif len(image.shape) == 3 and image.shape[2] == 4: # Handle RGBA if any
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            logging.debug("Input image is 4-channel BGRA. Converted to RGB for ONNX model input.")
        else: # If it's grayscale or some other format, log a warning and proceed without conversion
            logging.warning(f"Input image has {image.shape[2] if len(image.shape) == 3 else 'unknown'} channels. "
                            "ONNX models typically expect 3 (RGB) channels. Proceeding without specific conversion.")
            img = image 

        # Resize the image to the model's expected input dimensions (640x640 for YOLOv11 nano).
        img_resized = cv2.resize(img, (640, 640))
        # Normalize pixel values to the 0-1 range and convert to float32
        img_input = img_resized.astype(np.float32) / 255.0
        # Transpose image dimensions from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        img_input = np.transpose(img_input, (2, 0, 1))  
        # Add a batch dimension at the beginning (Batch, Channels, Height, Width)
        img_input = np.expand_dims(img_input, axis=0)   

        # Get the input name required by the ONNX model (usually 'images' for Ultralytics models)
        input_name = self.ort_session.get_inputs()[0].name
        logging.debug(f"ONNX input name identified: '{input_name}'")
        logging.debug(f"Sending input to ONNX model: shape={img_input.shape}, dtype={img_input.dtype}")

        # --- Run Inference ---
        # Perform the actual forward pass through the ONNX model
        outputs = self.ort_session.run(None, {input_name: img_input})
        
        # --- Post-processing ONNX Output ---
        # The raw output from the ONNX model needs specific steps to be turned into usable detections.
        # For this YOLOv11 ONNX export, the raw output shape is typically (1, 5, 8400)
        # where 5 attributes are (x, y, w, h, objectness_score), and 8400 are raw predictions.
        
        raw_predictions = outputs[0] # Get the primary output array from the model
        logging.debug(f"Raw ONNX model output shape received: {raw_predictions.shape}")

        # Transpose the output to make it easier to work with.
        # From (batch_size, attributes, num_predictions) to (batch_size, num_predictions, attributes)
        # So (1, 5, 8400) becomes (1, 8400, 5)
        raw_predictions = np.transpose(raw_predictions, (0, 2, 1)) 
        
        # Now, prepare the data for the Non-Maximum Suppression (NMS) function.
        # The NMS function expects a specific format: (..., x, y, w, h, obj_conf, class_prob, class_id)
        # Since our model is single-class, the objectness_score (index 4) serves as the class probability.
        
        obj_conf = raw_predictions[..., 4:5] # Extract objectness score (shape: (1, 8400, 1))
        
        # Concatenate raw bounding box coordinates (x,y,w,h), objectness_score,
        # objectness_score again (as class probability), and class ID (0 for golf_ball).
        # This results in a shape like (1, 8400, 7)
        combined_for_nms = np.concatenate((raw_predictions[..., :5], obj_conf, np.zeros_like(obj_conf)), axis=-1)

        # Apply Non-Maximum Suppression to filter and clean up detections.
        # This will return the best, non-overlapping boxes.
        final_predictions_list = non_max_suppression(
            combined_for_nms,
            conf_thres=self.confidence_threshold,
            iou_thres=0.45, # Common IoU threshold for NMS
            classes=[0],    # Only consider detections for class ID 0 ('golf_ball')
            multi_label=False # False because we're not doing multi-label classification per box
        )
        
        # The NMS function returns a list of arrays, one array per image in the batch.
        # Since we only process one image at a time, we take the first (and only) element.
        final_predictions = final_predictions_list[0] 
        logging.debug(f"Shape of predictions after NMS: {final_predictions.shape}")

        # --- Scale Detections Back to Original Frame Size ---
        # The ONNX model works with 640x640 images. The detected box coordinates
        # are normalized (0-1 range) relative to that size after NMS.
        # We need to scale them back to the original video frame's dimensions (height h0, width w0).
        h0, w0 = image.shape[:2] # Get original image height and width
        
        for det in final_predictions:
            # Each detection 'det' from NMS is in (x1, y1, x2, y2, confidence, class_id) format,
            # and these coordinates are NORMALIZED (0-1 range).
            x1_norm, y1_norm, x2_norm, y2_norm = det[:4] 
            conf = det[4]
            cls = int(det[5])

            logging.debug(f"Pre-scaling normalized coordinates (from NMS): x1={x1_norm:.4f}, y1={y1_norm:.4f}, x2={x2_norm:.4f}, y2={y2_norm:.4f}")

            # Scale normalized coordinates directly to the original image's pixel dimensions.
            x1_scaled = int(x1_norm * w0) # Scale x-coordinates by original image width
            y1_scaled = int(y1_norm * h0) # Scale y-coordinates by original image height
            x2_scaled = int(x2_norm * w0)
            y2_scaled = int(y2_norm * h0)

            # Calculate width and height of the scaled bounding box
            w_scaled = x2_scaled - x1_scaled
            h_scaled = y2_scaled - y1_scaled

            # Calculate the center point of the scaled box
            center_x = x1_scaled + w_scaled // 2
            center_y = y1_scaled + h_scaled // 2

            # Store the detection details in a dictionary
            detection_info = {
                'bbox': [x1_scaled, y1_scaled, w_scaled, h_scaled], # Stored as [x_top_left, y_top_left, width, height]
                'confidence': float(conf),
                'center': (center_x, center_y),
                'class_id': cls,
                'class_name': self.class_names[cls] if cls < len(self.class_names) else f"Unknown_Class_{cls}"
            }
            detections.append(detection_info)
            logging.info(f"Accepted detection: {detection_info}")

        logging.info(f"Total valid detections for current frame: {len(detections)}")
        return detections

    @property
    def input_shape(self):
        # This property is generally not used by ONNX Runtime for inference,
        # but kept for compatibility with earlier versions of your script.
        return (640, 640, 3) # (Height, Width, Channels) for the model's typical input


# --- The Main Script Execution (where everything kicks off!) ---
def main():
    # Set up argument parsing to allow running the script with command-line options
    parser = argparse.ArgumentParser(description="Run YOLO Ball Detection on videos, images, or webcam.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to your trained ONNX model file (e.g., 'runs/detect/train/weights/best.onnx'). This is a mandatory argument!")
    parser.add_argument('--video', type=str, help="Path to an input video file (e.g., 'my_clip.mp4').")
    parser.add_argument('--webcam', type=int, default=None,
                        help="Webcam ID (e.g., 0 for the default webcam). If set, it takes precedence over --video.")
    parser.add_argument('--image', type=str, help="Path to a single image file for detection.")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save the output video or image. Defaults to 'output_video_detected.mp4' or 'output_image_detected.jpg'.")
    parser.add_argument('--conf', type=float, default=0.5,
                        help="Confidence threshold (0.0 to 1.0). Detections with lower confidence are ignored. Default is 0.5.")

    args = parser.parse_args()

    logging.info("============================================================")
    logging.info("YOLO BALL DETECTION SCRIPT - Starting Up!")
    logging.info("============================================================")
    logging.info(f"Command-line arguments received: {vars(args)}")

    try:
        # Initialize the BallDetector with the specified model and confidence
        logging.info("Attempting to load the ball detector model...")
        detector = BallDetector(model_path=args.model, confidence_threshold=args.conf)
        logging.info(f"Model loaded successfully from {args.model}.")

        # --- Trajectory/Tracer State (for video only) ---
        trajectory = deque(maxlen=100)  # Store up to 100 points
        impact_detected = False
        last_center = None
        impact_distance_thresh = 20  # pixels, adjust as needed

        # For extrapolation when ball is lost
        extrapolation_frames = 0
        max_extrapolation = 100  # Only extrapolate for up to 100 frames after losing the ball
        arc_generated = False

        # --- Process a single image if --image argument is provided ---
        if args.image:
            logging.info(f"Processing a single image: '{args.image}'.")
            if not Path(args.image).is_file():
                logging.critical(f"Error: Image file not found at: '{args.image}'. Please verify the path.")
                raise FileNotFoundError(f"Image file not found: {args.image}")
            
            image_frame = cv2.imread(args.image)
            if image_frame is None:
                logging.critical(f"Error: Could not read image from '{args.image}'. Is it a valid image file or corrupted?")
                raise IOError(f"Failed to load image: {args.image}")

            detections = detector.detect_balls(image_frame.copy()) # Run detection on a copy of the image
            
            # Draw detections onto a copy of the image frame
            output_image_frame = image_frame.copy() 
            if detections:
                logging.info(f"Found {len(detections)} balls in the image. Drawing bounding boxes.")
                for det in detections:
                    x, y, w, h = det['bbox']
                    conf = det['confidence']
                    class_name = det['class_name']
                    center_x, center_y = det['center']
                    color = (0, 255, 0) # Green for the box
                    cv2.rectangle(output_image_frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(output_image_frame, label, (x, y - 10 if y - 10 > 0 else y + h + 20), # Adjust label position
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(output_image_frame, (center_x, center_y), 5, (0, 0, 255), -1) # Red dot for center
            else:
                logging.info("No balls detected in the image.")

            # Save the processed image to a file
            output_path = args.output if args.output else 'output_image_detected.jpg' 
            cv2.imwrite(output_path, output_image_frame)
            logging.info(f"Processed image saved to: {output_path}")
            
            # Display the image until a key is pressed (only works on systems with a display)
            cv2.imshow('Ball Detection - Image', output_image_frame)
            logging.info("Image displayed. Press any key on the window to close it.")
            cv2.waitKey(0)
            cv.destroyAllWindows()
            exit(0) # Exit the script after processing the image

        # --- If no image, then we're dealing with video or webcam input ---
        cap: Optional[cv2.VideoCapture] = None # OpenCV video capture object
        source_name: str = "" # A friendly name for the input source

        # Determine if it's a webcam or a video file
        if args.webcam is not None:
            logging.info(f"Attempting to open webcam with ID: {args.webcam}...")
            cap = cv2.VideoCapture(args.webcam)
            source_name = f"Webcam {args.webcam}"
        elif args.video:
            logging.info(f"Attempting to open video file: '{args.video}'...")
            if not Path(args.video).is_file(): # Ensure the video file exists
                logging.critical(f"Error: Video file not found at: '{args.video}'. Please verify the path.")
                raise FileNotFoundError(f"Video file not found: {args.video}")
            cap = cv2.VideoCapture(args.video)
            source_name = f"Video: '{args.video}'"
        else:
            logging.critical("Error: No video source specified. Please use --video or --webcam arguments.")
            raise ValueError("No video source specified.")

        # Check if the video source was opened successfully
        if not cap.isOpened():
            logging.critical(f"Error: Could not open video source: {source_name}. Check device/path and permissions.")
            raise IOError(f"Could not open video source: {source_name}")

        # Get properties of the video stream for setting up the output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Video source '{source_name}' opened. Dimensions: {frame_width}x{frame_height}, FPS: {fps:.2f}")

        # Setup VideoWriter object to save the processed video
        output_video_path = args.output if args.output else 'output_video_detected.mp4' # Default output name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files (common and widely supported)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        logging.info(f"Output video will be saved to: '{output_video_path}'.")
        logging.info("Starting video processing. Press 'q' key on the video window to quit early.")

        frame_count = 0
        while True:
            ret, frame = cap.read() # Read a frame from the video stream
            if not ret: # 'ret' is False if no more frames or an error occurred
                logging.info("End of video stream or failed to read frame. Finishing video processing.")
                break

            frame_count += 1
            logging.debug(f"Processing frame {frame_count}...")

            # Perform ball detection on the current frame
            detections = detector.detect_balls(frame.copy()) # Pass a copy to avoid modifying original frame

            # --- Ball Tracing Logic ---
            # Only consider the most confident detection (if any)
            if detections:
                # Sort detections by confidence, descending
                detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
                main_det = detections[0]
                center = main_det['center']

                # Detect impact (ball starts moving)
                if last_center is not None:
                    dist = np.linalg.norm(np.array(center) - np.array(last_center))
                    if not impact_detected and dist > impact_distance_thresh:
                        impact_detected = True
                        logging.info(f"Impact detected at frame {frame_count} (distance moved: {dist:.1f})")
                last_center = center

                # If impact detected, add to trajectory
                if impact_detected:
                    trajectory.append(center)
                    extrapolation_frames = 0  # Reset extrapolation when ball is found
                    arc_generated = False  # Reset arc flag if ball is found again
            else:
                # No detection, do not update last_center
                # If impact detected and we have at least 2 points, extrapolate
                frame_h, frame_w = frame.shape[:2]
                if impact_detected and len(trajectory) >= 2 and extrapolation_frames < max_extrapolation and not arc_generated:
                    pt1 = np.array(trajectory[-2])
                    pt2 = np.array(trajectory[-1])
                    velocity = pt2 - pt1
                    extrapolated = (pt2 + velocity).astype(int)
                    # Check if extrapolated point is out of bounds
                    out_of_bounds = not (0 <= extrapolated[0] < frame_w and 0 <= extrapolated[1] < frame_h)
                    if extrapolation_frames > 20 or out_of_bounds:
                        # Generate a parabolic arc for landing
                        arc_points = []
                        arc_len = 30  # Number of points in the arc
                        start = pt2
                        # End point: x continues, y is bottom of frame
                        end_x = int(pt2[0] + velocity[0] * 1.5)
                        end_x = max(0, min(end_x, frame_w - 1))
                        end_y = frame_h - 1
                        for t in np.linspace(0, 1, arc_len):
                            # Parabola: interpolate x, y with a curve
                            x = int((1 - t) * start[0] + t * end_x)
                            # Parabolic y: start to end_y, with a curve
                            y = int((1 - t) * start[1] + t * end_y - 0.25 * np.sin(np.pi * t) * abs(velocity[1]))
                            y = min(y, frame_h - 1)
                            arc_points.append((x, y))
                        trajectory.extend(arc_points)
                        arc_generated = True
                        logging.info(f"Landing arc generated at frame {frame_count}.")
                    else:
                        trajectory.append(tuple(extrapolated))
                        last_center = tuple(extrapolated)
                        extrapolation_frames += 1
                # If not enough points or max extrapolation reached or arc already generated, do not update trajectory

            # --- Draw Detections on the Frame ---
            for det in detections:
                x, y, w, h = det['bbox']
                conf = det['confidence']
                class_name = det['class_name']
                center_x, center_y = det['center']

                color = (0, 255, 0) # Green color for bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # Draw the bounding box
                label = f"{class_name}: {conf:.2f}" # Create text label (e.g., "golf_ball: 0.95")
                cv2.putText(frame, label, (x, y - 10 if y - 10 > 0 else y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1) # Draw a red dot at the center

            # --- Draw Tracer (Trajectory) ---
            if impact_detected and len(trajectory) > 1:
                tracer_color = (255, 0, 0)  # Blue
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i-1]
                    pt2 = trajectory[i]
                    cv2.line(frame, pt1, pt2, tracer_color, 2)
                    cv2.circle(frame, pt1, 2, tracer_color, -1)
                    cv2.circle(frame, pt2, 2, tracer_color, -1)

            # Display the processed frame in a window (for real-time viewing)
            cv2.imshow('Ball Detection Feed', frame)
            
            # Write the current processed frame to the output video file
            out.write(frame)

            # Check for 'q' key press to quit the video playback loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User pressed 'q' key. Stopping video playback.")
                break

    # --- Error Handling Blocks ---
    # Catch specific exceptions for clearer error messages
    except FileNotFoundError as fnfe:
        logging.critical(f"Error: A required file was not found: {fnfe}. Please double-check your model, video, or image paths.")
    except IOError as ioe:
        logging.critical(f"Error: Input/Output operation failed: {ioe}. This could be an issue with opening video/webcam, or saving the output file (permissions?).")
    except ValueError as ve:
        logging.critical(f"Error: Configuration issue: {ve}. Please review your command-line arguments or the 'class_names' setting in the script.")
    except Exception as e:
        # A general catch-all for any other unexpected errors.
        # exc_info=True will print the full Python traceback, which is very helpful for debugging.
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

    finally:
        # --- Resource Cleanup (important to release camera, video files, and windows) ---
        if 'cap' in locals() and cap is not None and cap.isOpened():
            cap.release() # Release the video capture object
            logging.info("Video capture released.")
        if 'out' in locals() and out is not None and out.isOpened():
            out.release() # Release the video writer object
            logging.info(f"Output video saved to '{output_video_path}'.")
        cv2.destroyAllWindows() # Close all OpenCV display windows
        logging.info("All OpenCV windows closed. Script execution finished.")


if __name__ == "__main__":
    main() # Call the main function to start the script!