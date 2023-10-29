import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from inference import Detect
from fastapi.responses import Response

from PIL import Image
import io

app = FastAPI()


def process_frame(frame_data):
    pil_image = Detect(frame_data)  # You should already have the PIL image here
    # Convert the PIL image to bytes
    with io.BytesIO() as output:
        pil_image.save(output, format="JPG")  # You can specify the desired format
        image_bytes = output.getvalue()
    return image_bytes


@app.post("/process_frame/")
async def upload_frame(file: UploadFile):
    if file.content_type and file.content_type.startswith("image/"):
        frame_data = await file.read()
        image_bytes = process_frame(frame_data)
        return Response(
            content=image_bytes, media_type="image/jpg"
        )  # Adjust media_type as needed


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
