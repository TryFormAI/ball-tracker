import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import time
from ball_detector import BallDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BallTracker:
    
    def __init__(self, 
                 detector_model_path: Optional[str] = None,
                 max_trajectory_length: int = 30,
                 min_detection_confidence: float = 0.5,
                 max_ball_distance: int = 100):
        
        self.detector = BallDetector(detector_model_path)
        self.max_trajectory_length = max_trajectory_length
        self.min_detection_confidence = min_detection_confidence
        self.max_ball_distance = max_ball_distance
        
        # Tracking state
        self.trajectories = []  # List of trajectory points for each ball
        self.ball_states = []   # List of current ball states
        self.frame_count = 0
        self.impact_detected = False
        
        # Colors for visualization
        self.ball_color = (0, 255, 0)  # Green for ball
        self.trajectory_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 0),    # Green
            (255, 255, 0)   # Cyan
        ]
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return it with ball tracking and tracers"""
        
        self.frame_count += 1
        
        # Detect balls in current frame
        detections = self.detector.detect_balls(frame)
        
        # Update tracking state
        self._update_tracking(detections)
        
        # Draw results on frame
        output_frame = self._draw_tracking_results(frame)
        
        return output_frame
    
    def _update_tracking(self, detections: List[Dict]):
        
        current_balls = []
        
        for detection in detections:
            if detection['confidence'] < self.min_detection_confidence:
                continue
            
            ball_center = detection['center']
            ball_radius = max(detection['bbox'][2], detection['bbox'][3]) // 2
            
            # Try to match with existing balls
            matched = False
            for i, existing_ball in enumerate(self.ball_states):
                distance = np.sqrt(
                    (ball_center[0] - existing_ball['center'][0])**2 +
                    (ball_center[1] - existing_ball['center'][1])**2
                )
                
                if distance < self.max_ball_distance:
                    # Update existing ball
                    old_center = existing_ball['center']
                    existing_ball['center'] = ball_center
                    existing_ball['radius'] = ball_radius
                    existing_ball['confidence'] = detection['confidence']
                    existing_ball['last_seen'] = self.frame_count
                    
                    # Check for impact (significant movement)
                    if not self.impact_detected and distance > 20:
                        self.impact_detected = True
                        logging.info(f"Impact detected at frame {self.frame_count}")
                    
                    # Add to trajectory if impact detected
                    if self.impact_detected:
                        self.trajectories[i].append({
                            'center': ball_center,
                            'frame': self.frame_count,
                            'confidence': detection['confidence']
                        })
                        
                        # Limit trajectory length
                        if len(self.trajectories[i]) > self.max_trajectory_length:
                            self.trajectories[i].pop(0)
                    
                    current_balls.append(existing_ball)
                    matched = True
                    break
            
            if not matched:
                # New ball detected
                new_ball = {
                    'center': ball_center,
                    'radius': ball_radius,
                    'confidence': detection['confidence'],
                    'last_seen': self.frame_count,
                    'id': len(self.ball_states)
                }
                
                self.ball_states.append(new_ball)
                self.trajectories.append([])
                current_balls.append(new_ball)
        
        # Remove balls that haven't been seen for a while
        self._cleanup_old_balls()
        
        # Update ball states
        self.ball_states = current_balls
    
    def _cleanup_old_balls(self):
        current_time = self.frame_count
        max_frames_missing = 10
        
        # Remove old balls and their trajectories
        indices_to_remove = []
        for i, ball in enumerate(self.ball_states):
            if current_time - ball['last_seen'] > max_frames_missing:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            self.ball_states.pop(i)
            if i < len(self.trajectories):
                self.trajectories.pop(i)
    
    def _draw_tracking_results(self, frame: np.ndarray) -> np.ndarray:
        
        output_frame = frame.copy()
        
        # Draw trajectories
        for i, trajectory in enumerate(self.trajectories):
            if len(trajectory) < 2:
                continue
            
            # Choose color for this trajectory
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            
            # Draw trajectory line
            for j in range(1, len(trajectory)):
                pt1 = trajectory[j-1]['center']
                pt2 = trajectory[j]['center']
                
                # Draw line with thickness based on confidence
                thickness = max(1, int(trajectory[j]['confidence'] * 5))
                cv2.line(output_frame, pt1, pt2, color, thickness)
                
                # Draw small circles at trajectory points
                cv2.circle(output_frame, pt1, 2, color, -1)
                cv2.circle(output_frame, pt2, 2, color, -1)
        
        # Draw current ball detections
        for i, ball in enumerate(self.ball_states):
            center = ball['center']
            radius = ball['radius']
            confidence = ball['confidence']
            
            # Draw ball circle
            cv2.circle(output_frame, center, radius, self.ball_color, 2)
            
            # Draw confidence text
            conf_text = f"{confidence:.2f}"
            cv2.putText(output_frame, conf_text, 
                       (center[0] + radius + 5, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ball_color, 1)
            
            # Draw ball ID
            cv2.putText(output_frame, f"Ball {ball['id']}", 
                       (center[0] - radius - 30, center[1] - radius - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ball_color, 1)
        
        # Draw frame info
        cv2.putText(output_frame, f"Frame: {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Balls: {len(self.ball_states)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Impact: {self.impact_detected}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write frame if output requested
            if writer:
                writer.write(processed_frame)
            
            # Display frame
            cv2.imshow('Ball Tracking', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Save current frame
                cv2.imwrite(f'balltracker/frame_{frame_count:04d}.jpg', processed_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed
                progress = frame_count / total_frames * 100
                logging.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_processed:.1f} fps")
        
        # Cleaning up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        logging.info(f"Processing complete. Processed {frame_count} frames.")
    
    def get_trajectory_data(self) -> List[List[Dict]]:
        """Get trajectory data for analysis"""
        return self.trajectories
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.trajectories = []
        self.ball_states = []
        self.frame_count = 0
        self.impact_detected = False

if __name__ == "__main__":
    tracker = BallTracker()
    

    # Add your own implementation here
    tracker.process_video('balltracker/video.mp4', 'balltracker/output.mp4')
    