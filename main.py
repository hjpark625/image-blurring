from fastapi import FastAPI, File, UploadFile
from image_blur import blurring_image, ResponseDTO

app = FastAPI()


@app.get("/")
async def root():
    return ResponseDTO(message="server is running")


# TODO: 전반적으로 사람의 얼굴과 번호판의 각도, 사이즈가 다양할때 블러처리가 제대로 되지 않음, 사람은 다수일때 블러처리가 제대로 되지 않음
@app.post("/image")
async def process_image(image: UploadFile = File(...)):
    contents = await image.read()
    origin_file_name = image.filename

    result = await blurring_image(contents, origin_file_name)

    return result


@app.post("/images")
async def multiple_images(images: list[UploadFile] = File(...)):
    for image in images:
        contents = await image.read()
        origin_file_name = image.filename
        result = await blurring_image(contents, origin_file_name)

    return result
