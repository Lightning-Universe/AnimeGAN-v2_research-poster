import logging
from functools import partial

import gradio as gr
import lightning as L
import requests
import torch
from PIL import Image
from lightning.components.serve import ServeGradio
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


# Credit to @akhaliq for his inspiring work.
# Find his original code there: https://huggingface.co/spaces/akhaliq/AnimeGANv2/blob/main/app.py
class ModelDemo(ServeGradio):
    inputs = gr.inputs.Image(type="pil", label="Upload to Animate your photo")
    outputs = gr.outputs.Image(type="pil", label="Animated Output")

    # elon = "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg"
    # img = Image.open(requests.get(elon, stream=True).raw)
    # img.save('elon.jpg')
    # examples = [['elon.jpg']]

    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute("gpu", 1))

    def predict(self, img):
        return self.model(img=img)

    def build_model(self):
        repo = "AK391/animegan2-pytorch:main"
        model = torch.hub.load(repo, "generator", device="cpu")
        face2paint = torch.hub.load(repo, "face2paint", size=512, device="cpu")
        return partial(face2paint, model=model)
