from fastapi import FastAPI, Query
from typing import List
from starlette.responses import JSONResponse
import torch
import pandas as pd
import numpy as np
from joblib import load

import sys
sys.path.insert(1, '..')
from pytorch import PytorchMultiClass
model=PytorchMultiClass(num_features=5)
model.load_state_dict(torch.load("../models/pytorch_multi_beer.pt"))

app = FastAPI()
sc=load("../models/sc_model.joblib")
le=load("../models/le_model.joblib")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/health/', status_code=200)
def healthcheck():
    return 'Neural Network model is ready to predict!'


def format_features(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv:float):
    return {
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }



@app.get("/beer/type/")
def predict(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv:float):
    features = format_features(brewery_name,review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    df=pd.DataFrame(features)
    df.drop('brewery_name',axis=1,inplace=True)
    num_cols=['review_aroma','review_appearance','review_palate','review_taste','beer_abv']
    df[num_cols]=sc.transform(df[num_cols])
    obs = torch.Tensor(df.to_numpy())
    pred_index= model(obs)[0].argmax(0)
    pred= le.inverse_transform([np.array(pred_index)])
    return JSONResponse(pred.tolist())


def format_features_multi(
    brewery_name: List[str]=Query(default=...), 
    review_aroma: List[float]=Query(default=...), 
    review_appearance: List[float]=Query(default=...), 
    review_palate: List[float]=Query(default=...), 
    review_taste: List[float]=Query(default=...), 
    beer_abv:List[float]=Query(default=...)):

    return {
        'brewery_name': brewery_name,
        'review_aroma': review_aroma,
        'review_appearance': review_appearance,
        'review_palate': review_palate,
        'review_taste': review_taste,
        'beer_abv': beer_abv
    }
    

@app.get("/beer/types/")
def predict(
    brewery_name: List[str]=Query(default=...), 
    review_aroma: List[float]=Query(default=...), 
    review_appearance: List[float]=Query(default=...), 
    review_palate: List[float]=Query(default=...), 
    review_taste: List[float]=Query(default=...), 
    beer_abv:List[float]=Query(default=...)):
    
    features = format_features_multi(brewery_name,review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    df=pd.DataFrame.from_dict(features)
    df.drop('brewery_name',axis=1,inplace=True)
    num_cols=['review_aroma','review_appearance','review_palate','review_taste','beer_abv']
    df[num_cols]=sc.transform(df[num_cols])
    pred=[]
    for i in range(0,df.shape[0]):
        obs = torch.Tensor(df.iloc[[i]].to_numpy())
        y_index= model(obs)[0].argmax(0)
        y= le.inverse_transform([np.array(y_index)])[0]    
        pred.append(y)   
    return JSONResponse(pred)



