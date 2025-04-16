import cv2
from utils import helpers


def viola_jones_face_detect(image_path: str, show: bool=True):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
    # mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    img = cv2.imread(image_path.decode() if isinstance(image_path, bytes) else image_path)
    if img is None:
        return 0
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 8)

        # eyes = eye_cascade.detectMultiScale(gray_img)
        # for (ex, ey, ew, eh) in eyes:
        #     center = (ex + ew, ey + eh)
        #     cv2.circle(gray_img, center, 3, (255, 0, 0), 5)

        # nose = nose_cascade.detectMultiScale(gray_img, 1.1, 5)
        # for (nx, ny, nw, nh) in nose:
        #     center = (x + nx + nw // 2, y + ny + nh // 2)
        #     cv2.circle(gray_img, center, 3, (255, 0, 0), 5)
        #     break  # Only mark one nose

        # mouths = mouth_cascade.detectMultiScale(gray_img)
        # for (mx, my, mw, mh) in mouths:
        #     center = (mx + mw, my + mh)
        #     cv2.circle(gray_img, center, 3, (255, 0, 0), 5)
        #     break  # Only mark one mouth

    if show:
        cv2.namedWindow("Detected Faces", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Detected Faces", cv2.cvtColor(helpers.calculate_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(faces)

def viola_jones_face_detect_webcam():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        
        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Webcam Face Detection", helpers.calculate_frame(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

