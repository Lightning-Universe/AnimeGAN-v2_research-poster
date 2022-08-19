# ⚡️ Anime Converter (AnimeGANV2 research poster)

This app is a research poster demo of AnimeGANv2 paper. It showcasese the paper, blog, noteboo,k and model demo where
you can upload an image and convert it into Anime.
To create a research poster for your work
use the [Lightning Research Template app](https://github.com/PyTorchLightning/lightning-template-research-app).

**Important Links:**
* [Paper](https://link.springer.com/chapter/10.1007/978-981-15-5577-0_18)
* [Github official](https://github.com/TachibanaYoshino/AnimeGAN)
* [Github PyTorch Implementation](https://github.com/bryandlee/animegan2-pytorch)
* [Blog](https://tachibanayoshino.github.io/AnimeGANv2/)
* [Colab Notebook](https://drive.google.com/file/d/1PbBkmj1EhULvEE8AXr2z84pZ2DQJN4hc/view?usp=sharing)

## Getting started

To create a Research Poster you can install this app via the [Lightning CLI](https://lightning.ai/lightning-docs/) or
[use the template](https://docs.github.com/en/articles/creating-a-repository-from-a-template) from GitHub and
manually install the app as mentioned below.

### Installation

#### With Lightning CLI

`lightning install app lightning/anime-converter`


Once you have installed the app, you can goto the `research-poster-animeganv2` folder and
run `lightning run app app.py --cloud` from terminal.
This will launch the template app in your default browser with tabs containing research paper, blog, Training
logs, and Model Demo.

You should see something like this in your browser:

> ![image](./assets/demo.png)

You can modify the content of this app and customize it to your research.
At the root of this template, you will find [app.py](./app.py) that contains the `ResearchApp` class. This class
provides arguments like a link to a paper, a blog, and whether to launch a Gradio demo. You can read more about what
each of the arguments does in the docstrings.

### Highlights

- Provide the link for paper, blog, or training logger like WandB as an argument, and `ResearchApp` will create a tab
  for each.
- Make a poster for your research by editing the markdown file in the [resources](./resources/poster.md) folder.
- Add interactive model demo with Gradio app, update the gradio component present in the \[research_app (
  ./research_app/components/model_demo.py) folder.
- View a Jupyter Notebook or launch a fully-fledged notebook instance (Sharing a Jupyter Notebook instance can expose
  the cloud instance to security vulnerability.)
- Reorder the tab layout using the `tab_order` argument.

### Example

```python
# update app.py at the root of the repo
import lightning as L

paper = "https://arxiv.org/pdf/2103.00020.pdf"
blog = "https://openai.com/blog/clip/"
github = "https://github.com/mlfoundations/open_clip"
wandb = "https://wandb.ai/aniketmaurya/herbarium-2022/runs/2dvwrme5"

app = L.LightningApp(
    ResearchApp(
        poster_dir="resources",
        paper=paper,
        blog=blog,
        training_log_url=wandb,
        github=github,
        notebook_path="resources/Interacting_with_CLIP.ipynb",
        launch_gradio=True,
    )
)
```
