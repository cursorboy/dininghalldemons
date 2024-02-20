import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import time

# COCO class names mapping
COCO_NAMES = {1: "person"}

def detect_objects(frame, model, device):
    # Convert frame to the format the model expects
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = F.to_tensor(frame_rgb).to(device)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        prediction = model(image)
    
    return prediction[0]

def main(video_stream_url):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    cap = cv2.VideoCapture(video_stream_url)
    frame_count = 0
    fps = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        predictions = detect_objects(frame, model, device)

        # Filter for the person class and draw bounding boxes
        for i, box in enumerate(predictions['boxes']):
            score = predictions['scores'][i]
            label_id = predictions['labels'][i].item()
            if score > 0.5 and label_id == 1:  # Check for 'person' class and high confidence
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                label_name = COCO_NAMES.get(label_id, 'Unknown')

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate and display FPS
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_stream_url = 'https://api.ucsb.edu/dining/cams/v2/stream/de-la-guerra?ucsb-api-key=0AVpBy0HfloWaEQHPanHTGSYmXusaNIJ'
    main(video_stream_url)



