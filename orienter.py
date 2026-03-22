import os
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from typing import Optional, Tuple, Union, List
import time
from pathlib import Path
import logging
import traceback

log = logging.getLogger(__name__)
current_dir = Path(__file__).parent

class Orienter:
    """
    Page orientation correction using EAST text detection and Hough transforms.
    Supports GPU acceleration and batch processing.
    """
    
    def __init__(
        self,
        east_model_path: Optional[str] = None,
        angle_tolerance: float = 0.25,
        min_confidence: float = 0.5,
        margin_tolerance: int = 9,
        east_width: int = 1280,
        east_height: int = 1280,
        use_gpu: bool = True,
        auto_detect_gpu: bool = True
    ):
        """
        Initialize the Orienter with configuration parameters.
        
        Args:
            east_model_path: Path to EAST text detection model
            angle_tolerance: Minimum angle threshold for rotation
            min_confidence: Minimum confidence for text detection
            margin_tolerance: Margin for angle filtering
            east_width: Width for EAST model input
            east_height: Height for EAST model input
            use_gpu: Whether to use GPU acceleration
            auto_detect_gpu: Automatically detect GPU availability
        """
        self.angle_tolerance = angle_tolerance
        self.min_confidence = min_confidence
        self.margin_tolerance = margin_tolerance
        self.east_width = east_width
        self.east_height = east_height
        
        # Set model path
        if east_model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.east_model_path = os.path.join(current_dir, "frozen_east_text_detection.pb")
        else:
            self.east_model_path = east_model_path
        
        # Load model once during initialization
        self.net = cv2.dnn.readNet(self.east_model_path)
        
        # Configure GPU support
        self.use_gpu = self._configure_gpu(use_gpu, auto_detect_gpu)
        
        # Layer names for EAST model
        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]
    
    def _configure_gpu(self, use_gpu: bool, auto_detect: bool) -> bool:
        """
        Configure GPU acceleration for OpenCV DNN module.
        
        Args:
            use_gpu: User preference for GPU usage
            auto_detect: Whether to auto-detect GPU availability
            
        Returns:
            bool: True if GPU is configured and available
        """
        if not use_gpu:
            return False
        
        if auto_detect:
            try:
                # Check if CUDA is available
                cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if cuda_available:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    log.info(f"GPU acceleration enabled. CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
                    return True
                else:
                    log.info("CUDA not available. Using CPU.")
                    return False
            except AttributeError:
                log.warning("OpenCV not compiled with CUDA support. Using CPU.")
                return False
        else:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                log.info("GPU acceleration manually enabled.")
                return True
            except Exception as e:
                log.error(f"Failed to enable GPU: {e}. Using CPU.")
                log.error(traceback.format_exc())
                return False
    
    def _rotate_bound(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle while maintaining full image content.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return cv2.warpAffine(
            image, M, (nW, nH),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    
    def _east_detect(self, image: np.ndarray) -> float:
        """
        Detect text orientation using EAST text detector.
        
        Args:
            image: Input image
            
        Returns:
            Median angle of detected text regions
        """
        (H, W) = image.shape[:2]
        
        rW = W / float(self.east_width)
        rH = H / float(self.east_height)
        
        resized_image = cv2.resize(image, (self.east_width, self.east_height))
        
        blob = cv2.dnn.blobFromImage(
            resized_image, 1.0, (self.east_width, self.east_height),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layer_names)
        
        (numRows, numCols) = scores.shape[2:4]
        angles = []
        
        for y in range(numRows):
            scoresData = scores[0, 0, y]
            anglesData = geometry[0, 4, y]
            
            for x in range(numCols):
                if scoresData[x] < self.min_confidence:
                    continue
                
                angle = anglesData[x]
                angles.append(angle * 180 / np.pi)
        
        return np.median(angles) if angles else 0.0
    
    def _hough_transforms(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Hough line transform to detect dominant line orientations.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (accumulator, angles, distances)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = canny(thresh)
        
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)
        
        return accum, angles, dists
    
    def _east_hough_line(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Combine EAST detection with Hough transforms for robust angle detection.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (image, detected angle)
        """
        east_angle = self._east_detect(image)
        _, theta, _ = self._hough_transforms(image)
        
        theta_deg = np.rad2deg(np.pi / 2 - theta)
        
        low_thresh = east_angle - self.margin_tolerance
        high_thresh = east_angle + self.margin_tolerance
        
        filtered_theta = theta_deg[
            (theta_deg > low_thresh) & (theta_deg < high_thresh)
        ]
        
        final_angle = np.median(filtered_theta) if len(filtered_theta) > 0 else east_angle
        
        return image, final_angle
    
    def re_orient_east(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Reorient a single image using EAST text detection and Hough transforms.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (corrected image, detected angle)
        """
        image, angle = self._east_hough_line(image)
        
        if abs(angle) > self.angle_tolerance:
            image = self._rotate_bound(image, angle)
        
        return image, angle
    
    def batch_reorient(
        self,
        images: Union[List[np.ndarray], List[str]],
        return_angles: bool = False,
        verbose: bool = True
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        """
        Batch process multiple images for orientation correction.
        
        Args:
            images: List of images (numpy arrays) or image file paths
            return_angles: Whether to return detected angles
            verbose: Print progress information
            
        Returns:
            List of corrected images, optionally with list of angles
        """
        corrected_images = []
        detected_angles = []
        
        total = len(images)
        start_time = time.time()
        
        for idx, img_input in enumerate(images):
            # Load image if path is provided
            if isinstance(img_input, str):
                image = cv2.imread(img_input)
                if image is None:
                    log.warning(f"Failed to load image: {img_input}")
                    corrected_images.append(None)
                    detected_angles.append(None)
                    continue
            else:
                image = img_input
            
            # Process image
            corrected_img, angle = self.re_orient_east(image)
            corrected_images.append(corrected_img)
            detected_angles.append(angle)
            
            if verbose and (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (total - idx - 1)
                log.info(f"Processed {idx + 1}/{total} images. \nAvg: {avg_time:.3f}s/img. ETA: {remaining:.1f}s")
        
        if verbose:
            total_time = time.time() - start_time
            log.info(f"Batch processing completed. \nTotal time: {total_time:.2f}s. \nAvg: {total_time/total:.3f}s/img")
        
        if return_angles:
            return corrected_images, detected_angles
        return corrected_images

