from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from os import name
import pathlib
from fastbook import load_learner, Path
from uvicorn import run

templates = Jinja2Templates(directory = "templates")

if name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
learn = load_learner(Path('./model.pkl'))

app = FastAPI(
    title="Covid Classification API",
    description="This API is part of Chayoot Kosiwanich's final project under the supervision of the AI Builders camp.",
    version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.get("/doc", response_class=HTMLResponse)
async def doc(request: Request):
    return templates.TemplateResponse("doc.html", {"request" : request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.filename[-4:] in ['.jpg', '.png', 'jpeg']:
        pred, pred_idx, probs = learn.predict(file.file.read())
        return {"filename" : file.filename,
                "prediction" : pred,
                "probability" : round(float(probs[pred_idx]), 4)}

    else: return {"your request is denied. support .jpg, .png, .jpec file."}

@app.get("/form")
async def main(request: Request):
    return templates.TemplateResponse("form.html", {"request" : request})

@app.post("/result")
async def predict(request: Request, file: UploadFile = File(...)):

    if file.filename[-4:] in ['.jpg', '.png', 'jpeg']:
        pred, pred_idx, probs = learn.predict(file.file.read())
        return templates.TemplateResponse("result.html", {"request" : request, 
                                                          "fname" : file.filename,
                                                          "pred" : pred,
                                                          "prob" : round(float(probs[pred_idx])*100, 4)})

    elif file.filename == "":
        return templates.TemplateResponse("form.html", {"request" : request})

    else: return {"your request is denied. support .jpg, .png, .jpec file."}
    
if __name__ == "__main__":
    run(app)