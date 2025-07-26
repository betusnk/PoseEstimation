import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.1) as pose:
    while cap.isOpened():
        ret, Image = cap.read()
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        Image.flags.writeable = False
        results = pose.process(Image)
        Image.flags.writeable = True
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)  # Re-enable color conversion
        mp_drawing.draw_landmarks(Image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=8, circle_radius=1),
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=1))
        cv2.imshow('deneme', Image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()