import time
from ultralytics import YOLO
import cv2


video_path = 'crowd.mp4'  
video_path_out = 'crowd_out.mp4'  

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Ошибка: Не удалось загрузить видео.")
    exit()

H, W, _ = frame.shape
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

model = YOLO('yolov8n.pt') 

threshold = 0.3 

total_people = 0
frame_count = 0


start_time = time.time()

while ret:
    frame_count += 1

    results = model(frame)[0]

    people_count = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and results.names[int(class_id)] == 'person':
            people_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    total_people += people_count
    cv2.putText(frame, f'People Count: {people_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

    ret, frame = cap.read()

cap.release()
out.release()

end_time = time.time()
processing_time = end_time - start_time
print(f"Время обработки: {processing_time:.2f} секунд")
print(f"Обработано кадров: {frame_count}")
