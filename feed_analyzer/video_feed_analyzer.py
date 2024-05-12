from ultralytics import YOLO
from datetime import datetime
import cv2
import os
import time
import pygame


def play_sound(sound_file):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


occurrences_for_class = {key: 0 for key in range(9)}
accuracy_threshold = 0.6
occurence_threshold = 5
video_save_interval = 300
bird_repellent_sound = 'metal_pipe.mp3'
bird_dict = {
    0: 'pigeon',
    1: 'crow',
    2: 'wood-pigeon',
    3: 'dove',
    4: 'magpie',
    5: 'jay',
    6: 'tit',
    7: 'thrush'
}
log_file = "bird_detection_log.txt"

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'detection_recordings/output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

if not cap.isOpened():
    exit()

os.makedirs('detection_photos', exist_ok=True)
os.makedirs('detection_recordings', exist_ok=True)
start_time = time.time()
last_video_save_time = time.time()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame, verbose=False)[0]

        for row in result.boxes.data:
            pred = row[4]
            if pred.numel() < 1:
                continue
            if pred >= accuracy_threshold:
                class_id = row[5].item()
                occurrences_for_class[class_id] += 1

        annotated_frame = result.plot()
        out.write(annotated_frame)
        cv2.imshow('Camera Feed', annotated_frame)

        if time.time() - start_time >= 1:
            for i in range(9):
                if occurrences_for_class[i] >= occurence_threshold:
                    with open(log_file, "a") as f:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"{current_time}: {bird_dict[i]}\n")

            if occurrences_for_class[0] >= occurence_threshold or occurrences_for_class[1] >= occurence_threshold:
                print(f'{occurrences_for_class[0] + occurrences_for_class[1]} invasive species detected in last second')
                play_sound(bird_repellent_sound)

            occurrences_for_class = {key: 0 for key in range(9)}
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_video_duration = time.time() - last_video_save_time

        if current_video_duration >= video_save_interval:
            out.release()
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
            video_filename = f'detection_recordings/output_{current_time}.mp4'
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
            last_video_save_time = time.time()

except Exception as e:
    print("An error occurred:", e)

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()