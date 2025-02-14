import cv2
import mediapipe as mp
import csv
import time
import signal
import sys

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

csv_file = open('landmarks.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp'] + [f'Landmark_{i}_x' for i in range(468)] + [f'Landmark_{i}_y' for i in range(468)])

def signal_handler(sig, frame):
    print("Ctrl+C pressed. Closing video and saving...")
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            timestamp = time.time() - start_time
            landmark_row = [timestamp]
            for landmark in face_landmarks.landmark:
                landmark_row.extend([landmark.x, landmark.y])
            csv_writer.writerow(landmark_row)

    out.write(frame)

    cv2.imshow('Face Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
