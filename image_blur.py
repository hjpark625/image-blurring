import cv2
import numpy as np
from pathlib import Path


def decode_image(image_data: bytes):
    nparr = np.fromstring(image_data, np.uint8)
    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return decoded_image

async def blurring_image(image_data: bytes, origin_file_name: str):
    img = decode_image(image_data)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))
    
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in plates:
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))

    save_path = Path.home() / 'Desktop' / 'blurImages'
    save_path.mkdir(parents=True, exist_ok=True)

    print(img)
    
    cv2.imwrite(str(save_path / f'blur_{origin_file_name}'), img)

    return {'result': f'{save_path}/blur_{origin_file_name}에 저장됨'}
