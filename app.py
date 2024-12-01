from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import uvicorn
import os
from sklearn.linear_model import Ridge
import fastapi
from fastapi.responses import StreamingResponse
import io
from fastapi import File, UploadFile, HTTPException


app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

# Подгружаю pickle для своей лучшей модели
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
# Подгружаю pickle для ohe
ohe_path = os.path.join(os.path.dirname(__file__), "one_hot_enc.pkl")
with open(ohe_path, 'rb') as ohe_file:
    ohe = pickle.load(ohe_file)

# Функция, которая служит для предобработки исходных данных
def preprocess_data(data):
    data = data.drop(columns=['torque', 'selling_price', 'name'], axis=1)
    data['mileage'] = data['mileage'].apply(lambda x: float(str(x).split(' ')[0]) if x != None else None)
    data['engine'] = data['engine'].apply(lambda x: float(str(x).split(' ')[0]) if x != None else None)
    data['max_power'] = data['max_power'].apply(lambda x: float(str(x).split(' ')[0]) if x != None else None)
    data['seats'] = data['seats'].astype(int)
    data['seats'] = data['seats'].astype(str)

    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

    encoded = ohe.transform(data[cat_features]).toarray()
    encoded_df = pd.DataFrame(encoded,
                              columns=ohe.get_feature_names_out(cat_features),
                              index=data.index)

    car_encoded = pd.concat([data.drop(columns=cat_features), encoded_df], axis=1)
    return car_encoded#car_encoded

@app.get("/")
async def root():
    return {
        "Name": "Car price prediction",
        "Description": "This is a model for predicting the price of a car based on its characteristics.."
    }

# Делаем предсказание для одной машины и выводим целевую переменную -
@app.post("/predict_item")
def predict_item(item: Item):# -> float:
    item = item.dict()
    item = pd.DataFrame(item, index=[0])
    processed_data = preprocess_data(item)
    prediction = loaded_model.predict(processed_data)
    return round(float(prediction[0]), 2)

# Делаем предсказания для списка машин и выгружаем csv
@app.post("/predict_items/")
def predict_items(file: UploadFile = File(...)):
    try:
        data = file.file.read()
        df = pd.read_csv(io.BytesIO(data))

        processed_data = preprocess_data(df)
        predictions = loaded_model.predict(processed_data)
        df['predicted_price'] = predictions

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()