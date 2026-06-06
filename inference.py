import cv2
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F

from models.slowfast_model import SuspiciousActivityModel
from utils import pack_pathway_output

def main():
    # ============================================================
    # Device Configuration and Setup
    # ============================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Live Stream Inference on: {DEVICE}")

    NUM_FRAMES = 32
    IMG_SIZE = 224
    
    # DISPLAY CONFIGURATION: Set the maximum width of the screen
    # The height will scale automatically to maintain the correct aspect ratio.
    MAX_DISPLAY_WIDTH = 1024  
    
    INFERENCE_FREQUENCY = 2  
    frame_count = 0

    CLASSES = [
        "Normal", 
        "Violence", 
        "Theft", 
        "Property Damage",
        "Harassment"
    ]

    # ============================================================
    # Load the SLOWFAST brain
    # ============================================================
    print("Loading Trained SlowFast Architecture...")
    model = SuspiciousActivityModel(num_classes=len(CLASSES))
    
    checkpoint_path = "checkpoints/slowfast_dcsass_e8.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Successfully loaded weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Weights not found at {checkpoint_path}.")
        return
        
    model.to(DEVICE).eval()

    # ============================================================
    # Stream initialization and buffer
    # ============================================================
    # STREAM_SOURCE = "car_stealing.mp4" # Set to 0 for webcam, or your video string path
    # STREAM_SOURCE = "barber.mp4"
    # STREAM_SOURCE = "shoplifting.mp4"
    STREAM_SOURCE = "beating.mp4"
    # STREAM_SOURCE = "bar_shooting.mp4"
    cap = cv2.VideoCapture(STREAM_SOURCE)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open stream source: {STREAM_SOURCE}")
        return

    frame_buffer = deque(maxlen=NUM_FRAMES)
    current_prediction = "Buffering Stream Layers..."
    confidence_pct = 0.0

    print("\nCCTV Engine Initialized. Press 'q' to safely terminate stream feed.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break

        frame_count += 1

        # ====================================================
        # AI spatial preprocessing (Keeps 224x224 for Brain)
        # ====================================================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_ai_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        normalized_frame = resized_ai_frame.astype(np.float32) / 255.0
        
        tensor_frame = torch.tensor(normalized_frame).permute(2, 0, 1)
        frame_buffer.append(tensor_frame)

        # ====================================================
        # Cognitive inference window
        # ====================================================
        if len(frame_buffer) == NUM_FRAMES and (frame_count % INFERENCE_FREQUENCY == 0):
            video_tensor = torch.stack(list(frame_buffer), dim=1)
            video_tensor = video_tensor.unsqueeze(0).to(DEVICE)

            inputs = pack_pathway_output(video_tensor)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                    
                confidence, predicted = torch.max(probabilities, 1)
                current_prediction = CLASSES[predicted.item()]
                confidence_pct = confidence.item() * 100

        # Dynamic screen resize
        orig_h, orig_w = frame.shape[:2]
        
        # Calculate scaling scale factors based on target monitor width
        if orig_w > MAX_DISPLAY_WIDTH:
            scale_factor = MAX_DISPLAY_WIDTH / orig_w
            display_w = MAX_DISPLAY_WIDTH
            display_h = int(orig_h * scale_factor)
        else:
            display_w = orig_w
            display_h = orig_h

        # Downsample the view frame cleanly for screen rendering
        display_frame = cv2.resize(frame, (display_w, display_h))

        # ====================================================
        # Security Overlay (Rendered on resized window)
        # ====================================================
        is_buffered = len(frame_buffer) == NUM_FRAMES
        status_color = (0, 0, 255) if (current_prediction != "Normal" and is_buffered) else (0, 255, 0)
        
        if not is_buffered:
            status_color = (0, 255, 255) 
            display_text = f"SYSTEM WARNING: Buffering Stream ({len(frame_buffer)}/{NUM_FRAMES})"
        else:
            display_text = f"ALERT STATUS: {current_prediction} [{confidence_pct:.1f}%]"

        # Dark status banner
        cv2.rectangle(display_frame, (0, 0), (display_w, 50), (15, 15, 15), -1)
        cv2.putText(display_frame, display_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        # Show the window
        cv2.imshow("Unified SlowFast CCTV Feed Analyzer", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()