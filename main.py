import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

app = FastAPI()
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

@app.get('/')
async def root():
    return {'message': 'server is running'}

@app.post('/image')
async def process_image(image: UploadFile = File(...)):
    contents = await image.read()
    
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 얼굴 인식 및 블러처리
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))

    # 번호판 인식 및 블러처리 얼굴 로직과 유사함
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in plates:
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))

    temp_file = NamedTemporaryFile(delete=True, suffix='.jpg')
    file_name = temp_file.name.split('/')[-1]
    save_path = f'{desktop_path}/blurImages'

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    cv2.imwrite(f'{save_path}/${file_name}', img)
    return {'result': f'{save_path}에 저장됨'}

@app.post('/images')
async def multiple_images(images: list[UploadFile] = File(...)):
    for image in images:
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        for (x, y, w, h) in faces:
            img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))
        
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        plates = plate_cascade.detectMultiScale(img, 1.1, 4)
        for (x, y, w, h) in plates:
            img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))
        
        temp_file = NamedTemporaryFile(delete=True, suffix='.jpg')
        file_name = temp_file.name.split('/')[-1]
        save_path = f'{desktop_path}/blurImages'

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        cv2.imwrite(f'{save_path}/{file_name}', img)

    return {'result': f'{save_path}에 저장됨'}