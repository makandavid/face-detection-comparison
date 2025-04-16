import cv2
from screeninfo import get_monitors

def calculate_scale(image: cv2.typing.MatLike):
    monitor = get_monitors()[0]
    return min(monitor.width / image.shape[1], monitor.height / image.shape[0])

def calculate_frame(image: cv2.typing.MatLike):
    scale = calculate_scale(image)
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)