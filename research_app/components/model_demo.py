from functools import partial

import gradio as gr
import requests
import torch
from PIL import Image
from lightning.app.components.serve import ServeGradio
import io


# Credit to @akhaliq for his inspiring work.
# Find his original code there: https://huggingface.co/spaces/akhaliq/AnimeGANv2/blob/main/app.py
class ModelDemo(ServeGradio):
    inputs = gr.inputs.Image(type="pil", label="Upload to Animate your photo")
    outputs = gr.outputs.Image(type="pil", label="Animated Output")
    enable_queue = True

    elon = "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg"
    r = requests.get(elon, stream=True)
    img = Image.open(io.BytesIO(r.content))
    img.save('elon.jpg')

    examples = [['elon.jpg']]

    def __init__(self):
        super().__init__()

    def predict(self, img):
        return self.model(img=img)

    def build_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        repo = "AK391/animegan2-pytorch:main"
        model = torch.hub.load(repo, "generator", device=device)
        face2paint = torch.hub.load(repo, "face2paint", size=512, device=device)
        return partial(face2paint, model=model)
