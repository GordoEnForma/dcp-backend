from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from numpy import array
from uvicorn import run
import numpy as np

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

model_dir = "model.h5"
model = tf.keras.models.load_model(model_dir)

class_predictions = array(["Queratosis actínicas","carcinoma de células basales","lesiones benignas similares a queratosis","dermatofibroma ", "melanoma", "nevo melanocítico", "vascular lesions"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}

@app.post("/predict")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    
    img_path = tf.keras.utils.get_file(
        origin = image_link
    )
    img = tf.keras.utils.load_img(
        img_path, 
        target_size = (256, 256)
    )
    
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    print(predictions)
    
    labels = ["Queratosis actínicas","carcinoma de células basales","lesiones benignas similares a queratosis","dermatofibroma ", "melanoma", "nevo melanocítico", "vascular lesions"]
    prediction_labels = [labels[i] for i in np.argmax(predictions, axis=1)]

    return {
        "filename": image_link,
        "prediction": prediction_labels[0],
        "probability": predictions[0][np.argmax(predictions, axis=1)][0]*100
    }

if __name__ == "__main__":
    run(app)