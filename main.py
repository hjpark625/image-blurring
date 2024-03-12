from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

app = FastAPI()

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
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), thickness=-1)

    # 번호판 인식 및 블러처리 얼굴 로직과 유사함
    # TODO: 번호판 인식 모델이 러시아 번호판 모델인데 한국 번호판 모델이 따로 필요할지 생각해보기
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in plates:
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), thickness=-1)
    
    # 처리된 이미지 저장 및 변환
    
    # print(save_path)

    temp_file = NamedTemporaryFile(delete=True, suffix='.jpg')
    file_name = temp_file.name.split('/')[-1]
    save_path = f'/Users/jeff/Desktop/{file_name}'

    cv2.imwrite(save_path, img)
    return FileResponse(path=save_path, media_type='image/jpeg')

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
        save_path = f'/Users/jeff/Desktop/{file_name}'
        cv2.imwrite(save_path, img)

    return FileResponse(path=save_path, media_type='image/jpeg')