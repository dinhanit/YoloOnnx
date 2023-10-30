import numpy as np
from fastapi import FastAPI, UploadFile
from io import BytesIO
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from inference import Inference
IF = Inference()

app = FastAPI()

def process(img_np):
    return IF.Detect(img_np)

@app.post("/process_frame/")
async def upload_frame(file: UploadFile):
    if file.content_type and file.content_type.startswith("image/"):
        frame_data = await file.read()
        image_pil = Image.open(BytesIO(frame_data))
        image_np = np.array(image_pil)
        processed_img = process(image_np)
        print(type(processed_img))
        img_bytes = BytesIO()
        processed_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return StreamingResponse(io.BytesIO(img_bytes.read()), media_type="image/jpeg")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8501)
  