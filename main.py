#main.py(modified)

import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys
import uvicorn

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi import FastAPI, status, UploadFile, File


from ultralytics import YOLO

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection using YOLOv8 and FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.6.1",
)

# This function is needed if you want to allow client requests
# from specific domains (specified in the origins argument)
# to access resources from the FastAPI server,
# and the client and server are hosted on different domains.

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# No need to include the detection_post router

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation
    data of the FastAPI application to a JSON file.
    The purpose of saving the OpenAPI documentation data is to have
    a permanent and offline record of the API specification,
    which can be used for documentation purposes or
    to generate client libraries. It is not necessarily needed,
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False,  tags=['docs'])
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK, tags=['Health Check'])
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}

########################Loading my Model################################
model = YOLO('best.pt')

######################### Support Func #################################

def predict_image(image: Image) -> dict:
    # Make a prediction
    results = model.predict(source=image)

    # Dictionary to store counts of each object
    object_counts = {"can": 0, "bottle": 0}

    # Interpret the prediction
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                label = box.cls.cpu().numpy()[0]  # Move tensor to CPU before converting to NumPy array
                confidence = box.conf.cpu().numpy()[0]  # Move tensor to CPU before converting to NumPy array
                # Assuming class 39 is 'bottle' and class 0 is 'can'
                if label == 39:
                    logger.info(f"The model predicts: Bottle with confidence {confidence:.2f}")
                    object_counts["bottle"] += 1
                else:
                    logger.info(f"The model predicts: Unidentified Object with confidence {confidence:.2f}")
                    object_counts["can"] += 1

    return object_counts

# Route to receive image file and perform object detection
@app.post("/detect-objects/", response_model=dict, tags=["Object Detection"])
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(file.file)

        # Predict objects in the image
        object_counts = predict_image(image)

        # Log the object counts
        logger.info(f"Object Counts: {object_counts}")

        return object_counts
    except Exception as e:
        logger.exception("Error occurred during object detection.")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "_main_":
    uvicorn.run(app, host='localhost', port=8000)