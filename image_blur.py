from pathlib import Path
import cv2
import numpy as np


def decode_image(image_data: bytes):
    nparr = np.fromstring(image_data, np.uint8)
    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return decoded_image


async def blurring_image(image_data: bytes, origin_file_name: str):
    img = decode_image(image_data)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=4)
    for x, y, w, h in faces:
        img[y : y + h, x : x + w] = cv2.blur(
            src=img[y : y + h, x : x + w], ksize=(30, 30)
        )

    plate_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    )
    plates = plate_cascade.detectMultiScale(img, 1.1, 4)
    for x, y, w, h in plates:
        img[y : y + h, x : x + w] = cv2.blur(img[y : y + h, x : x + w], (30, 30))

    save_path = Path.home() / "Desktop" / "blurImages"
    save_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_path / f"blur_{origin_file_name}"), img)

    return ResponseDTO(message=f"{save_path}/blur_{origin_file_name}에 저장됨")


class ResponseDTO:
    message: str

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message
