import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import datetime
import random
import folium
import geocoder
import firebase_admin
from firebase_admin import credentials, db

# ---------------------- FIREBASE SETUP ----------------------
cred = credentials.Certificate("C:\\F\\Sem-5\\Pothole-detection-yolo\\pothole-detection-yolo\\Implementation\\firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://pothole-detection-db-default-rtdb.asia-southeast1.firebasedatabase.app/"
})


def push_to_firebase(label, confidence, latitude, longitude, coords):
    ref = db.reference("potholes")
    ref.push({
        "label": label,
        "confidence": float(confidence),
        "latitude": latitude,
        "longitude": longitude,
        "bounding_box": coords,
        "timestamp": str(datetime.datetime.now())
    })

model = YOLO('C:\\F\\Sem-5\\Pothole-detection-yolo\\pothole-detection-yolo\\Implementation\\best.pt') 

def get_real_gps():
    try:
        g = geocoder.ip('me', timeout=5)  # set a 5-second timeout
        if g.ok and g.latlng:
            return g.latlng[0], g.latlng[1]
    except Exception as e:
        print("GPS error:", e)
    # fallback GPS if API fails
    return 17.384, 78.4564

def get_simulated_gps():
    latitude = 17.3850 + random.uniform(-0.01, 0.01)
    longitude = 78.4867 + random.uniform(-0.01, 0.01)
    return latitude, longitude


def plot_potholes_on_map():
    m = folium.Map(location=[17.3850, 78.4867], zoom_start=12)
    ref = db.reference("potholes")
    data = ref.get()

    if data:
        for key, row in data.items():
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=f'{row["label"]}, Conf: {row["confidence"]:.2f}'
            ).add_to(m)

    m.save("potholes_map.html")
    print("Map saved as potholes_map.html")

# --- Image Mode ---
def detect_potholes_image(image_path, save_output=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)


    annotated_image = results[0].plot()
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    if save_output:
        output_path = "detected_output.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated detection saved as {output_path}")

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls.item())]
            confidence = box.conf.item()
            coords = box.xyxy.tolist()

            lat, lon = get_simulated_gps()
            try:
                push_to_firebase(label, confidence, lat, lon, coords)
                print(f"Uploaded to Firebase → {label}, Conf: {confidence:.2f}, GPS: ({lat}, {lon})")
            except Exception as e:
                print("Firebase error:", e)

    plot_potholes_on_map()
    return results

# --- Real-Time Mode ---
def detect_potholes_realtime(conf_threshold=0.25, dark_threshold=30, max_dark_frames=15, save_snapshot=True):
    cap = cv2.VideoCapture(0)
    dark_frame_count = 0
    no_detection_count = 0
    snapshot_taken = False

    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    print("Starting real-time pothole detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.mean() < dark_threshold:
            dark_frame_count += 1
            print(f"⚠️ Camera dark for {dark_frame_count}/{max_dark_frames} frames")
            if dark_frame_count >= max_dark_frames:
                print("Camera dark for too long. Stopping detection.")
                break
            continue
        else:
            dark_frame_count = 0

        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        cv2.imshow("Pothole Detection (YOLOv8)", annotated_frame)

        if len(results[0].boxes) == 0:
            no_detection_count += 1
            if no_detection_count >= 20:
                print("No detections for 20 frames. Stopping detection...")
                break
        else:
            no_detection_count = 0
            for box in results[0].boxes:
                label = model.names[int(box.cls.item())]
                confidence = box.conf.item()
                coords = box.xyxy.tolist()
                lat, lon = get_real_gps()
                if lat is None or lon is None:
                    lat, lon = get_simulated_gps()
                push_to_firebase(label, confidence, lat, lon, coords)
                print(f"Uploaded → Label: {label}, Conf: {confidence:.2f}, GPS: ({lat}, {lon})")
                if save_snapshot and not snapshot_taken:
                    cv2.imwrite("detected_output.jpg", annotated_frame)
                    snapshot_taken = True
                    print("Snapshot saved as detected_output.jpg")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plot_potholes_on_map()


# --- Main selector ---
if __name__ == "__main__":
    mode = input("Choose mode: (1) Image mode, (2) Real-time mode: ")

    if mode == "1":
        test_image_path = input("Enter image path: ")
        detect_potholes_image(test_image_path)
    elif mode == "2":
        detect_potholes_realtime()
    else:
        print("Invalid choice. Please enter 1 or 2.")
