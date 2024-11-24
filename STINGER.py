import torch
import cv2
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# YOLOv5 Model Class
class YOLOModel:
    def __init__(self, model_path='ultralytics/yolov5'):
        # Load the YOLOv5 model from PyTorch Hub
        self.model = torch.hub.load(model_path, 'yolov5s', pretrained=True)
        self.model.eval()

    def detect_objects(self, frame):
        # Perform object detection on the frame
        results = self.model(frame)
        return results

# Llama 3.2 Model Class
class LlamaModel:
    def __init__(self, model_name='meta-llama/Llama-3.2-3B'):
        # Load the Llama 3.2 model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate_text(self, prompt):
        # Generate text based on the prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Vision-Language Integration Class
class VisionLanguageSystem:
    def __init__(self):
        self.yolo = YOLOModel()
        self.llama = LlamaModel()
        self.mission_active = False
        self.mission_type = None

    def process_frame(self, frame):
        # Detect objects in the frame
        results = self.yolo.detect_objects(frame)
        detected_objects = results.pandas().xyxy[0]['name'].tolist()

        # Generate description based on detected objects
        if detected_objects:
            prompt = f"In the current frame, I see the following objects: {', '.join(detected_objects)}. Describe the scene."
            description = self.llama.generate_text(prompt)
        else:
            description = "No objects detected in the current frame."

        return description, results

    def execute_mission(self, frame):
        # Execute mission based on the mission type
        if self.mission_type == 'surveillance':
            return self.surveillance_mission(frame)
        elif self.mission_type == 'search_and_rescue':
            return self.search_and_rescue_mission(frame)
        else:
            return "Unknown mission type."

    def surveillance_mission(self, frame):
        # Surveillance mission logic
        description, results = self.process_frame(frame)
        # Additional surveillance-specific processing can be added here
        return description

    def search_and_rescue_mission(self, frame):
        # Search and rescue mission logic
        description, results = self.process_frame(frame)
        # Additional search and rescue-specific processing can be added here
        return description

    def start_mission(self, mission_type):
        # Start a mission
        self.mission_active = True
        self.mission_type = mission_type
        print(f"Mission '{mission_type}' started.")

    def stop_mission(self):
        # Stop the current mission
        self.mission_active = False
        self.mission_type = None
        print("Mission stopped.")

# Command-Line Interface for Missions
def main():
    parser = argparse.ArgumentParser(description='Real-Time Vision-Language System with Mission Control')
    parser.add_argument('--mission', type=str, help='Type of mission to execute (e.g., surveillance, search_and_rescue)')
    args = parser.parse_args()

    system = VisionLanguageSystem()

    # Start video capture from the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    if args.mission:
        system.start_mission(args.mission)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            if system.mission_active:
                description = system.execute_mission(frame)
                print(description)

            # Display the frame with detections
            results = system.yolo.detect_objects(frame)
            results.render()
            cv2.imshow('Real-Time Object Detection', results.imgs[0])

            # Check for user input to stop the mission
            if cv2.waitKey(1) & 0xFF == ord('q'):
                system.stop_mission()
                break

            # Simulate mission duration
            time.sleep(0.1)  # Adjust as needed

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
