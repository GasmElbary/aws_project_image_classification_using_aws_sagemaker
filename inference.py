import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # calling gpu

def net():
    logger.info("Inside net()")
    model = models.efficientnet_b7(pretrained = True) 
    for param in model.parameters():
        param.requires_grad = False 
    
    num_features=model.classifier[1].in_features
    model.classifier[1] = nn.Sequential( nn.Linear(num_features, 256),
                             nn.ReLU(),
                             nn.Linear(256, 256),
                             nn.ReLU(),
                             nn.Linear(256, 128),
                             nn.ReLU(),
                             nn.Linear(128, 6),
                             nn.LogSoftmax(dim=1)
                            )
    return model

def model_fn(model_dir):
    logger.info("Inside model_fn()")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("loading the model...")
        model.load_state_dict(torch.load(f, map_location = device))
        logger.info("Model Loaded Success")
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info("Inside input_fn()")
    logger.info('Processing the input data.')

    logger.info(f'Incoming Requests Content-Type is: {content_type}')
    logger.info(f'Request body Type is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))


def predict_fn(input_object, model):
    logger.info("Inside predict_fn()")
    test_transform =  transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.ToTensor()]
                            )
 
    logger.info("Applying Transforms to inference image")
    input_object = test_transform(input_object)
	if torch.cuda.is_available():
        input_object = input_object.cuda()
    
    model.eval()
    with torch.no_grad():
        logger.info("Calling the model")
        prediction = model(input_object.unsqueeze(0))
    return prediction