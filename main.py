import cv2
import time
from camera import Camera
from detectors.face_recog import FaceRecog
from draw import draw_face

FACE_MODEL = "models/blaze_face_short_range.tflite"

def main():
    cam = Camera(index=0, width=1280, height=720)
    face = FaceRecog(FACE_MODEL)

    try:
        while True:
            frame = cam.capture()
            if frame is None:
                break
            faces = face.detect(frame)
            draw_face(frame, faces)

            cv2.imshow("Face and Gesture", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.release()
        face.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()