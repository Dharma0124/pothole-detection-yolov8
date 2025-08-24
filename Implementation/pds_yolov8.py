import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('C:\\F;\\Sem-5\\Software Engineering Lab(Team-8)\\Software Engineering Lab\\Implementation\\best.pt')  # Update the path to your .pt file

# Function to perform inference on a single test image
def detect_potholes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO
    
    # Perform inference using the YOLOv8 model
    results = model(image_rgb)  # YOLOv8 auto-scales and preprocesses the input
    
    # Visualize results
    annotated_image = results[0].plot()  # Annotate image with detections
    
    # Display the output image with pothole detection
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()
    
    # Extract and print detection details (bounding boxes, labels, confidence scores)
    for result in results:
        for box in result.boxes:
            print(f"Label: {model.names[box.cls.item()]}, Confidence: {box.conf.item()}, Coordinates: {box.xyxy.tolist()}")
    
    return results

# Example of testing on an image
test_image_path = 'C:\\F;\\Sem-5\\Software Engineering Lab(Team-8)\\Software Engineering Lab\\Implementation\\valid images\\download (7).jpeg'  # Update the path to your test image
detect_potholes(test_image_path)
