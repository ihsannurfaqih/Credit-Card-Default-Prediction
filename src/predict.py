import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn

class Customer(BaseModel):
    PAY_AMT6: float = Field(..., examples=9905)
    PAY_AMT5: float = Field(..., examples=54379)
    PAY_AMT4: float = Field(..., examples=201)
    PAY_AMT3: float = Field(..., examples=49450)
    PAY_AMT2: float = Field(..., examples=48276)
    PAY_AMT1: float = Field(..., examples=99159)
    BILL_AMT6: float = Field(..., examples=54109)
    BILL_AMT5: float = Field(..., examples=40018)
    BILL_AMT4: float = Field(..., examples=49403)
    BILL_AMT3: float = Field(..., examples=48017)
    BILL_AMT2: float = Field(..., examples=97882)
    BILL_AMT1: float = Field(..., examples=33179)
    PAY_6: int = Field(..., examples=-1)
    PAY_5: int = Field(..., examples=0)
    PAY_4: int = Field(..., examples=-1)
    PAY_3: int = Field(..., examples=-1)
    PAY_2: int = Field(..., examples=-1)
    PAY_0: int = Field(..., examples=-1)

class PredictResponse(BaseModel):
    default_probability: float
    default: bool


app = FastAPI(title="customer-default-prediction")

with open('final_model.pkl', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        default_probability=prob,
        default=bool(prob >= 0.52)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)