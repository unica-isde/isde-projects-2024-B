import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images
from app.forms.histogram_form import HistogramForm
from app.forms.transformation_form import TransformationForm
from app.ml.histogram_utils import histogram
from app.ml.transformation_utils import transform_image


app = FastAPI()
config = Configuration()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/info")
def info() -> dict[str, list[str]]:
    """Returns a dictionary with the list of models and
    the list of available image files."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    data = {"models": list_of_models, "images": list_of_images}
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service."""
    return templates.TemplateResponse("home.html", {"request": request, "active_page": "home"})


@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models, "active_page": "classifications"},
    )


@app.post("/classifications")
async def request_classification(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
            "active_page": "classifications",
        },
    )


@app.get("/histogram")
def create_histogram(request: Request):
    return templates.TemplateResponse(
        "histogram_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models, "active_page": "histogram"},
    )


@app.post("/histogram")
async def request_histogram(request: Request):
    form = HistogramForm(request)
    await form.load_data()
    image_id = form.image_id
    hist_b, hist_g, hist_r = histogram(image_id)
    return templates.TemplateResponse(
        "histogram_output.html",
        {
            "request": request,
            "image_id": image_id,
            "histogram_blue": json.dumps(hist_b),
            "histogram_green": json.dumps(hist_g),
            "histogram_red": json.dumps(hist_r),
            "active_page": "histogram",
        },
    )

@app.get("/transformation")
def create_transformation(request: Request):
    return templates.TemplateResponse(
        "transformation_select.html",
        {
            "request": request,
            "images": list_images(),
            "models": Configuration.models,
            "active_page": "transformation",
        },
    )

@app.post("/transformation")
async def request_transformation(request: Request):
    form = TransformationForm(request)
    await form.load_data()
    image_id = form.image_id
    color = form.color
    brightness = form.brightness
    contrast = form.contrast
    sharpness = form.sharpness
    transform_image(image_id, color, brightness, contrast, sharpness)
    return templates.TemplateResponse(
        "transformation_output.html",
        {
            "request": request,
            "image_id": image_id,
            "active_page": "transformation",
        },
    )