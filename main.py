import json
import os
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
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
from app.ml.upload_utils import uploaded_image
from fastapi.responses import JSONResponse, FileResponse
import matplotlib.pyplot as plt
import io
import asyncio


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
    return templates.TemplateResponse(
        "home.html", {"request": request, "active_page": "home"}
    )


@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {
            "request": request,
            "images": list_images(),
            "models": Configuration.models,
            "active_page": "classifications",
        },
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
        {
            "request": request,
            "images": list_images(),
            "models": Configuration.models,
            "active_page": "histogram",
        },
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
async def request_transformation(request: Request, background_tasks: BackgroundTasks):
    form = TransformationForm(request)
    await form.load_data()
    image_id = form.image_id
    color = form.color
    brightness = form.brightness
    contrast = form.contrast
    sharpness = form.sharpness
    image_enhanced_id=transform_image(image_id, color, brightness, contrast, sharpness)

    # Delete the file after 30 seconds
    image_enhanced_path=f"app/static/enhanced_images/{image_enhanced_id}"
    background_tasks.add_task(delete_file_after_delay, image_enhanced_path)

    return templates.TemplateResponse(
        "transformation_output.html",
        {
            "request": request,
            "image_id": image_id,
            "image_enhanced_id": image_enhanced_id,
            "active_page": "transformation",
        },
    )

@app.get("/download_json")
async def download_json(classification_scores: str):
    try:

        scores = json.loads(classification_scores)

        return JSONResponse(
            content=scores,
            headers={
                "Content-Disposition": "attachment; filename=classification_scores.json"
            },
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

@app.get("/download_graph")
async def download_graph(classification_scores: str):
    try:
        scores = json.loads(classification_scores)

        labels = [item[0] for item in scores]
        data = [item[1] for item in scores]

        plt.barh(
            labels, data, color=["#1a4a04", "#750014", "#795703", "#06216c", "#3f0355"]
        )
        plt.grid()
        plt.title("Classification Scores")

        plt.gca().invert_yaxis()

        img_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        plt.close()

        return Response(
            content=img_buf.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=classification_graph.png"
            },
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

@app.get("/upload")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "upload_select.html",
        {
            "request": request,
            "images": list_images(),
            "models": Configuration.models,
            "active_page": "upload",
        },
    )


@app.post("/upload")
async def classify_uploaded_image(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    uploaded_file = form.get("file_image")
    model_id = form.get("model_id")

    if not uploaded_file:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": "No file uploaded",
                "active_page": "upload",
            },
        )

    upload_dir = "app/static/uploads"
    os.makedirs(upload_dir, exist_ok=True)  # Crea la directory se non esiste
    file_location = f"{upload_dir}/{uploaded_file.filename}"

    file_data = await uploaded_file.read()

    with open(file_location, "wb") as output_file:
        output_file.write(file_data)

    classification_scores = uploaded_image(model_id=model_id, img_data=file_data)

    # Delete the file after 30 seconds
    background_tasks.add_task(delete_file_after_delay, file_location)

    return templates.TemplateResponse(
        "upload_output.html",
        {
            "request": request,
            "image_id": uploaded_file.filename,
            "classification_scores": json.dumps(classification_scores),
            "active_page": "upload",
        },
    )


async def delete_file_after_delay(filepath: str, delay: int = 30):
    await asyncio.sleep(delay)
    if os.path.exists(filepath):
        os.remove(filepath)

