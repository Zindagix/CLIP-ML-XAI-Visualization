from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import deploy_service as svc

app = FastAPI(title="CS5489 retrieval service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def check(tag):
    if tag not in svc.MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model {tag}")


@app.get("/models")
def models():
    return [svc.model_info(tag) for tag in svc.MODELS]


@app.get("/eval")
def evaluate(tag: str = "coco"):
    check(tag)
    try:
        return svc.evaluate(tag)
    except FileNotFoundError as err:
        raise HTTPException(status_code=409, detail=str(err))


@app.get("/text")
def text(query: str, tag: str = "coco", k: int = 8):
    check(tag)
    try:
        return svc.text_to_image(query, tag, k)
    except FileNotFoundError as err:
        raise HTTPException(status_code=409, detail=str(err))


@app.get("/image")
def image(image_id: int, tag: str = "coco", k: int = 5):
    check(tag)
    try:
        return svc.image_to_text(image_id, tag, k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"image {image_id} not in the test split")
    except FileNotFoundError as err:
        raise HTTPException(status_code=409, detail=str(err))


@app.get("/images")
def images(limit: int = 200):
    return svc.test_images(limit)
