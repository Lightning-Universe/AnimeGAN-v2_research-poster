import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)

from functools import partial
import gradio as gr
import torch
from PIL import Image
import requests
from lightning.components.serve import ServeGradio


# Credit to @akhaliq for his inspiring work.
# Find his original code there: https://huggingface.co/spaces/akhaliq/AnimeGANv2/blob/main/app.py
class ModelDemo(ServeGradio):
    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    elon = "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg"
    img = Image.open(requests.get(elon, stream=True).raw)
    img.save('elon.jpg')
    # examples = [['elon.jpg']]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, img):
        return self.model(img=img)

    def build_model(self):
        repo = "AK391/animegan2-pytorch:main"
        model = torch.hub.load(repo, "generator", device="cpu")
        face2paint = torch.hub.load(repo, "face2paint", size=512, device="cpu")
        self.ready = True
        return partial(face2paint, model=model)
