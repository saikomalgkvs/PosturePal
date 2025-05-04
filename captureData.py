import cv2
import mediapipe as mp
import pandas as pd
from timer import Timer

def capture_data(label, duration):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(0)  # Try changing 0 to 1 or 2 if you have multiple cameras

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None

    data = []
    start_time = Timer()
    start_time.start()

    while start_time.get_elapsed_time() < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            origin_x, origin_y, origin_z = nose.x, nose.y, nose.z

            row = {
                'left_ear_x': landmarks[mp_pose.PoseLandmark.LEFT_EAR].x - origin_x,
                'left_ear_y': landmarks[mp_pose.PoseLandmark.LEFT_EAR].y - origin_y,
                'left_ear_z': landmarks[mp_pose.PoseLandmark.LEFT_EAR].z - origin_z,
                'right_ear_x': landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x - origin_x,
                'right_ear_y': landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y - origin_y,
                'right_ear_z': landmarks[mp_pose.PoseLandmark.RIGHT_EAR].z - origin_z,
                'left_mouth_x': landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].x - origin_x,
                'left_mouth_y': landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].y - origin_y,
                'left_mouth_z': landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].z - origin_z,
                'right_mouth_x': landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].x - origin_x,
                'right_mouth_y': landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].y - origin_y,
                'right_mouth_z': landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].z - origin_z,
                'left_shoulder_x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - origin_x,
                'left_shoulder_y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y - origin_y,
                'left_shoulder_z': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z - origin_z,
                'right_shoulder_x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - origin_x,
                'right_shoulder_y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - origin_y,
                'right_shoulder_z': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z - origin_z,
            }
            data.append(row)

        # Display the webcam feed
        cv2.putText(frame, f"Recording {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {duration - start_time.get_elapsed_time():.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Posture Capture", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(data)
    # print(f"{label.capitalize()} posture data captured successfully.")
    return df

# Test the function
# capture_data("bad", 60)