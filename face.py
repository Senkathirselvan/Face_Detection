# face.py

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

class Face:
    def __init__(self, frame_names=None):
        self.frame_names = frame_names
        
        # Download model
        self.model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        print(f"Model downloaded to: {self.model_path}")

        # Load model
        self.model = YOLO(self.model_path)
        print("Model loaded successfully")

    def process_frames(self, frame_names):
        for img in frame_names:
            # Load image
            image_path = img
            print(f"Attempting to load image: {image_path}")
            image = Image.open(image_path)
            print(f"Image loaded successfully: {image_path}")

            # Perform inference
            output = self.model(image)
            print("Inference completed")

            # Convert results to Detections
            results = Detections.from_ultralytics(output[0])
            print("Detections converted")

            print(results)
