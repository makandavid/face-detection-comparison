from facenet_pytorch import MTCNN
import torch
import cv2
from PIL import Image
from utils import helpers

def init_mtcnn():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f'Running on device: {device}')
    return MTCNN(keep_all=True, device=device)
    
def mtcnn_face_detect(image_path, show: bool=True):
    mtcnn = init_mtcnn()
    img = cv2.imread(image_path.decode() if isinstance(image_path, bytes) else image_path)
    if img is None:
        return 0
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, _, landmarks = mtcnn.detect(rgb_img, landmarks=True)
    if boxes is None or landmarks is None:
        return 0
    
    for box, landmark_set in zip(boxes, landmarks):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

        for (x, y) in landmark_set:
            cv2.circle(rgb_img, (int(x), int(y)), 2, (255, 0, 0), 2)

    if show:
        cv2.namedWindow("Detected Faces", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Detected Faces", cv2.cvtColor(helpers.calculate_frame(rgb_img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(boxes)

    
def mtcnn_face_detect_webcam():
    mtcnn = init_mtcnn()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _, landmarks = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), landmarks=True)
        if boxes is not None and landmarks is not None:
            for box, landmark_set in zip(boxes, landmarks):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for (x, y) in landmark_set:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), 2)

        cv2.imshow("Webcam Face Detection", helpers.calculate_frame(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
