
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
from cv2 import cv2
import random
import os
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import _create_text_labels
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

#Setup arguments
parser = argparse.ArgumentParser(usage="run inference on one specified image")
parser.add_argument('--image', required=True, help="relative path to image to run inference on")
args = parser.parse_args()

#Register dataset annotations in coco format. This is important for metadata used by Detectron, for example in inference.
register_coco_instances("my_dataset_train", {}, "content/train/_annotations.coco.json", "content/train")
register_coco_instances("my_dataset_val", {}, "content/valid/_annotations.coco.json", "content/valid")

#Get metadata from training set, to provide metadata during inference. 
#This metadata allows the predictor to assign predictions to human-readable classes for output.
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

#setup config for inference.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12 #your number of classes + 1

# Inference with Detectron2 Saved Weights on a previously unknown image

# specify trained weights file
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the class confidence threshold for this model!
predictor = DefaultPredictor(cfg)


print("Starting to output inference results. Predictions below confidence score of 0.65 score ignored.")

#Initialise file for inference output data:
import json

#check if file has been written previously, and load it to be later merged with output data from new file
if os.path.exists('inferenceContent/outputData/outputData.json'):
    with open('inferenceContent/outputData/outputData.json', 'r') as jsonFile:
        dataCollection = json.load(jsonFile)

#if the file has not been yet initialised, or does not exist, write it with metadata to alter merge with output data from new file
else:
    dataCollection = {}
    AllClasses = my_dataset_train_metadata.thing_classes
    #add this to json object, if not there already.
    dataCollection['AllClasses'] = AllClasses
    #initialize empty dict to store image inference ouput results in the future
    dataCollection['imageData'] = {}
#Get all available classes for this dataset from training metadata. Needed to make counts for each class in inference results.


#Get name of uploaded image from argument. Nodejs launches shell script with filled argument. argument stored in imageName, similarly to standard batch inference.
imageName = args.image
if imageName:
    #Get uploaded image name without path so we can save to JSON conveniently.
    print("Running inference on image name with path: {}".format(imageName))
    imageBaseName = format(os.path.basename(imageName))
    print("Image base name: {}".format(imageBaseName))

    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    #Get class names from dataset train metadata to put on visualization.
                    metadata=my_dataset_train_metadata, 
                    scale=1
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #As described in draw_instance_predictions function of Visualizer, we can get data about each image. 
    #We might need these, but currently only interested in labels as they contain predicted classes for each image and prediction score.
    imagePredClasses = outputs["instances"].pred_classes
    imagePredBoxes = outputs["instances"].pred_boxes
    imagePredScores = outputs["instances"].scores

    #previously initialised dict for storing image output. Inside it make another key:value pair with key as image name, and value as inference output values array.
    dataCollection['imageData'][imageBaseName] = _create_text_labels(outputs["instances"].pred_classes, outputs["instances"].scores, my_dataset_train_metadata.get("thing_classes", None))

    
    if not os.path.exists('inferenceContent/outputData'):
        os.mkdir('inferenceContent/outputData')

    with open('inferenceContent/outputData/outputData.json', 'w') as jsonFile:
        json.dump(dataCollection, jsonFile)
    #data in it. As it loops through all images it will fill the JSON with needed data.

    # Show images with predictions in system window. If not using host, then:
    #cv2.imshow('Inference Preview',out.get_image()[:, :, ::-1])
    # If running inference locally, to view result with timer in system window, uncomment:
    #cv2.waitKey(10000)
    # Save images with predictions to savePath folder and imageName with path to image removed from name.
    savePath = './inferenceContent/output'
    cv2.imwrite(os.path.join(savePath , imageBaseName), out.get_image()[:, :, ::-1])