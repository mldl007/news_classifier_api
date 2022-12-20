from utils.load_model import load_model
import pandas as pd
import numpy as np
from logger.logger import MongoLogger


def predict(x: pd.Series, use_deployed_model: bool = True,
            model_file_name: str = None, predict_label: bool = False):
    x = x.copy()
    prediction = None
    prediction_proba = None
    label_encoder = None
    logger = MongoLogger()
    try:
        logger.log_to_db(level="INFO", message="entering model_inference.predict")

        loaded_model = load_model(load_deployed_model=use_deployed_model, model_file_name=model_file_name)

        if loaded_model is not None:
            (text_preprocess, vectorizer, dimensionality_reduction,
             minmax_scaler, label_encoder, classifier) = loaded_model

            # text preprocess
            preprocessed_x = text_preprocess.preprocess(string_series=x, dataset="test")
            # vectorization
            vectorized_x = vectorizer.vectorize(x=preprocessed_x, dataset="test")
            vectorized_x = pd.DataFrame(vectorized_x)
            # dimensionality reduction
            x = dimensionality_reduction.reduce_dimensions(vectorized_x, dataset="test")
            # minmax scaler
            if minmax_scaler is not None:
                x = minmax_scaler.transform(x).copy()

            prediction = (classifier.predict(x)).astype('int')
            prediction_proba = (classifier.predict_proba(x))
            if predict_label:
                prediction = label_encoder.inverse_transform(prediction)

    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected model_inference.predict error: {e}")
        raise
    logger.log_to_db(level="INFO", message="exiting model_inference.predict")

    return prediction, prediction_proba, label_encoder
