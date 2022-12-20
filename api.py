from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from pydantic import BaseModel
import pandas as pd
from model_inference.model_inference import predict
from logger.logger import MongoLogger
import traceback

app = FastAPI()


class Data(BaseModel):
    """
    Data dictionary for data type validation
    """
    text: str


@app.post("/")
def prediction(data: Data):
    """
    Processes the API request and returns a prediction
    """
    logger = MongoLogger()
    logger.log_to_db(level="INFO", message="entering prediction_api")
    try:
        df = pd.Series(data.dict())  # converting api data dict to df
        pred, prediction_proba, label_encoder = predict(df, predict_label=True)
        prediction_proba_list = prediction_proba.tolist()[0]
        prediction_proba_sorted = prediction_proba_list.copy()
        prediction_proba_sorted.sort(reverse=True)
        top_2_prediction_proba = prediction_proba_sorted[:2]
        result_list_labels = [label_encoder.inverse_transform([prediction_proba_list.index(pred)])[0]
                              for pred in top_2_prediction_proba]
        top_2_confidence_level = [np.round(pred * 100, 2) for pred in top_2_prediction_proba]
        response = {"result": result_list_labels, "confidence": top_2_confidence_level}
        json_compatible_item_data = jsonable_encoder(response)
        response = JSONResponse(content=json_compatible_item_data)

    except Exception as e:
        # executes in case of any exception
        pred = e
        logger.log_to_db(level="CRITICAL", message=f"unexpected error in prediction_api: {traceback.format_exc()}")
        raise
    logger.log_to_db(level="INFO", message="exiting prediction_api")
    return response


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5001)
