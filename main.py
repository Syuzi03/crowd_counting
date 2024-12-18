import time
from ultralytics import YOLO
import cv2


video_path = 'crowd.mp4'  
video_path_out = 'crowd_out.mp4'  

# capture video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Ошибка: Не удалось загрузить видео.")
    exit()

# Get videos frame and sizes 
H, W, _ = frame.shape
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# using YOLOv8 model
model = YOLO('yolov8n.pt') 

# confidence score
threshold = 0.3  # low score to detect small objects

# for counting frames and peopl
total_people = 0
frame_count = 0


start_time = time.time()

while ret:
    frame_count += 1

    # running current frame on model(YOLO)
    results = model(frame)[0]

    # results processing
    people_count = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # if confidence score > 0.3 and object is person:
        if score > threshold and results.names[int(class_id)] == 'person':
            people_count += 1
            # bounding boxes around object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # info about crowd count on current frame
    total_people += people_count
    cv2.putText(frame, f'People Count: {people_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # writng processed frame to new video
    out.write(frame)

    # get next frame
    ret, frame = cap.read()

# realese resources
cap.release()
out.release()

# check how long it took to process
end_time = time.time()
processing_time = end_time - start_time
print(f"Время обработки: {processing_time:.2f} секунд")
print(f"Обработано кадров: {frame_count}")
