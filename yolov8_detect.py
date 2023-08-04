import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')

# model = YOLO('pth/yolov8n.pt')

# model = YOLO('pths/yolov8/train/weights/best.pt')
model = YOLO('best.pt')
print('111')
# Open the video file
video_path = "72.mp4"
cap = cv2.VideoCapture(video_path)

detection_results = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, save_txt=True)

        # detection_results.append({
        #     'frame_id': len(detection_results),
        #     'results': results
        # })

        print(results[0])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


with open('detection_results.txt', 'w') as f:
    for frame_result in detection_results:
        frame_id = frame_result['frame_id']
        results = frame_result['results']

        for r in results:
            class_id, confidence, x1, y1, x2, y2 = r
            f.write(f"{frame_id} {class_id} {confidence} {x1} {y1} {x2} {y2}\n")

