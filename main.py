from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
from PIL import Image
from utils import translate_dease, get_advice, format_output, healthy_advice

import io


app = FastAPI()
model_classifier = load_learner('models/plant_disease_classifier.pkl', cpu=True)
model_filter = load_learner('models/plant_filter.pkl', cpu=True)

@app.post("/upload-image")
async def prediction(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {"status": "error", "message": f"Impossible de charger l'image: {str(e)}"}
    
    plant = model_filter.predict(img)[0]
    
    if plant != 'plant':
        return {"status": "error", "translation": "Ceci n'est pas une plante", "result": plant}
    
    prediction_disease = model_classifier.predict(img)
    desease = prediction_disease[0].replace('_', ' ')
    desease = format_output(desease).strip()
    confidence = prediction_disease[2].max().item()
    
    if desease.lower() == 'healthy':
        description = healthy_advice
    else:
        description = get_advice(desease=desease)['choices'][0]['message']['content']
    
    return {
        'status': 'success',
        'prediction': desease.strip(),
        'confidence': confidence,
        'translation': translate_dease(desease),
        'description': description,
    }
