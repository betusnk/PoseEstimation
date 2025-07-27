import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import time

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

VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions

base_options = BaseOptions(model_asset_path='pose_landmarker.task')
options = PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)
prev_frame_time = 0
new_frame_time = 0

count = 0
prev_fps = 0
font = cv2.FONT_HERSHEY_COMPLEX

while cap.isOpened():
  ret, image = cap.read()
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
  new_frame_time = time.time()
  fps = 1 / (new_frame_time - prev_frame_time)
  prev_frame_time = new_frame_time
  fps = int(fps)
  fpss = str(fps)
  count += 1
  if not count == 100:
      prev_fps = prev_fps + fps
  else:
      final_fps = prev_fps / 100
      print(final_fps)
      count = 0
      prev_fps = 0
  detection_result = detector.detect(image)
  image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  
  cv2.putText(image, fpss, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
  
  cv2.imshow('deneme', image)

  if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()