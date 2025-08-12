# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

#from predictor import VisualizationDemo
from predictor import DefaultPredictor
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def main():
    mp.set_start_method("spawn", force=True)

    args = get_parser().parse_args()
    
    # create the config
    cfg = setup_cfg(args)

    # Instantiate the predictor
    demo = DefaultPredictor(cfg)

    for path in tqdm.tqdm(args.input, disable=not args.output):
        
        img = read_image(path, format="BGR")
        predictions = demo(img)

        # save the predictions in a text file
        if len(predictions["instances"]) > 0:
            pred_classes = predictions["instances"].pred_classes.cpu().numpy()
            pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
            txt_filename = os.path.splitext(path)[0] + ".txt"
            with open(txt_filename, "w") as f:
                for cls, box in zip(pred_classes, pred_boxes):
                    f.write("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(cls, box[0], box[1], box[2], box[3]))
    
if __name__ == "__main__":
   main()