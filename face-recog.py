import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SPACING = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
RED_BGR = (0, 0, 255)

MODEL_PATH = "models/blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Could not open webcam")

detector = FaceDetector.create_from_options(options)

t0 = time.time()

while cam.isOpened():

    success, frame_bgr = cam.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    frame_timestamp_ms = int((time.time() - t0) * 1000)

    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    annotated_image = frame_bgr.copy()

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            start_point = (x, y)
            end_point = (x + w, y + h)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), RED_BGR, 2)

            if detection.categories:
                cat = detection.categories[0]
                name = cat.category_name or ""
                prob = f"{cat.score:.2f}"
                label = f"{name} ({prob})"
            else:
                label = "face"


            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE,
                FONT_THICKNESS,
            )
            text_location = (x + w // 2 - text_w // 2,
                             y + h + SPACING + text_h)
            cv2.putText(annotated_image, label, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, RED_BGR, FONT_THICKNESS)
            cv2.imshow("Face Detection", annotated_image)

    if cv2.waitKey(20) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()





