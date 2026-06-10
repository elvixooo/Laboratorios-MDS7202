import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="API Potabilidad del Agua", version="1.0")


class MedicionAgua(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


class RespuestaPotabilidad(BaseModel):
    potabilidad: int


@app.get("/")
def home():
    return {
        "modelo": "XGBoost Classifier",
        "problema": "Clasificación binaria de potabilidad del agua",
        "descripcion": (
            "Predice si el agua es potable (1) o no potable (0) a partir de 9 mediciones químicas de sensores IoT."
        ),
        "entrada": {
            "ph": "float — nivel de pH del agua",
            "Hardness": "float — dureza del agua (mg/L)",
            "Solids": "float — sólidos totales disueltos (ppm)",
            "Chloramines": "float — concentración de cloraminas (ppm)",
            "Sulfate": "float — concentración de sulfato (mg/L)",
            "Conductivity": "float — conductividad eléctrica (μS/cm)",
            "Organic_carbon": "float — carbono orgánico total (ppm)",
            "Trihalomethanes": "float — concentración de trihalometanos (μg/L)",
            "Turbidity": "float — turbidez del agua (NTU)",
        },
        "salida": {"potabilidad": "int — 1 si el agua es potable, 0 si no lo es"},
        "docs": "http://localhost:8000/docs",
    }


@app.post("/potabilidad/", response_model=RespuestaPotabilidad)
def predecir_potabilidad(medicion: MedicionAgua):
    data = pd.DataFrame([medicion.model_dump()])
    prediccion = model.predict(data)[0]
    return {"potabilidad": int(prediccion)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
