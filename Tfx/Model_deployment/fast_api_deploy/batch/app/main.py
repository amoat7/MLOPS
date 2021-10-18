# the server's code must be in the file main.py within a directory called app, following FastAPI's guidelines.

import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

app = FastAPI(title= "Predicting Wine Class")

#Now you need a way to represent a data point. You can do this by creating a class the subclasses from pydantic's BaseModel 
# and listing each attribute along with its corresponding type. 
# In this case a data point represents a wine so this class is called Wine and all of the features of the model are of type float

class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]
    

#Now it is time to load the classifier into memory so it can be used for prediction. 
# This can be done in the global scope of the script but here it is done inside a function to show you a cool feature of FastAPI.
#If you decorate a function with the @app.on_event("startup") decorator you ensure that the function is run at the startup of the server. 
# This gives you some flexibility if you need some custom logic to be triggered right when the server starts.
#The classifier is opened using a context manager and assigned to the clf variable, which you still need to make global so other functions can access it:

@app.on_event("startup")
def load_clf():
    # Load the classifier from the pickle file
    with open("/app/wine.pkl", "rb") as file:
        global clf 
        clf = pickle.load(file)

#Finally you need to create the function that will handle the prediction. 
# This function will be run when you visit the /predict endpoint of the server and it expects a Wine data point.
# This function is actually very straightforward, first you will convert the information within the Wine object into a numpy array of shape (1, 13) 
# #and then use the predict method of the classifier to make a prediction for the data point. Notice that the prediction must be casted into a list using the tolist method.
# Finally return a dictionary (which FastAPI will convert into JSON) containing the prediction.

@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}


'''
curl -X 'POST' http://localhost/predict \
  -H 'Content-Type: application/json' \
  -d '{
  "alcohol":12.6,
  "malic_acid":1.34,
  "ash":1.9,
  "alcalinity_of_ash":18.5,
  "magnesium":88.0,
  "total_phenols":1.45,
  "flavanoids":1.36,
  "nonflavanoid_phenols":0.29,
  "proanthocyanins":1.35,
  "color_intensity":2.45,
  "hue":1.04,
  "od280_od315_of_diluted_wines":2.77,
  "proline":562.0
}'


curl -X POST http://localhost:80/predict \
    -d @./wine-examples/1.json \
    -H "Content-Type: application/json"



'''
