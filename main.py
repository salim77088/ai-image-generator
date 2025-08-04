from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# حمل النموذج مرة وحدة
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# واجهة HTML بسيطة
@app.get("/", response_class=HTMLResponse)
def read_form():
    return """
    <html>
        <head>
            <title>مولد صور بالذكاء الاصطناعي</title>
        </head>
        <body style="text-align: center; font-family: Arial;">
            <h1>🖼️ مولد صور بالذكاء الاصطناعي</h1>
            <form action="/generate" method="post">
                <input type="text" name="prompt" placeholder="اكتب وصف الصورة" size="50" required>
                <br><br>
                <button type="submit">🎨 توليد</button>
            </form>
        </body>
    </html>
    """

# توليد صورة من الوصف
@app.post("/generate", response_class=HTMLResponse)
def generate_image(prompt: str = Form(...)):
    image = pipeline(prompt).images[0]
    filename = f"static/{uuid.uuid4().hex}.png"
    image.save(filename)
    return f"""
    <html>
        <body style="text-align: center; font-family: Arial;">
            <h2>✅ تم توليد الصورة!</h2>
            <img src="/{filename}" style="max-width: 90%; border-radius: 10px;"><br><br>
            <a href="/">🔙 رجوع</a>
        </body>
    </html>
    """

if not os.path.exists("static"):
    os.makedirs("static")
