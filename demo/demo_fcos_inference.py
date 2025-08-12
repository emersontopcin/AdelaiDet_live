# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

# Import detectron2
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import DefaultPredictor
from adet.config import get_cfg

# Import NuScenes
from nuscenes.nuscenes import NuScenes

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
        "--output-txt",
        type=str,
        default="/root/code/nuscenes/fcos/",
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
    parser.add_argument(
        "--data-root-nuscenes",
        type=str,
        default="/root/code/nuscenes/",
        help="Root directory of the NuScenes dataset.",
    )
    parser.add_argument(
        "--nuscenes-version",
        type=str,
        default="v1.0-mini",
        help="Version of the NuScenes dataset.",
    )
    return parser


def main():
    # Setup arguments
    args = get_parser().parse_args()

    # Garante que o diretório de saída existe
    os.makedirs(args.output_txt, exist_ok=True)

    # create the config
    cfg = setup_cfg(args)

    # Instantiate the predictor
    demo = DefaultPredictor(cfg)

    # Open the NuScenes dataset
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.data_root_nuscenes, verbose=True)

    # Get all keyframe front camera sample_data tokens
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['channel'] == 'CAM_FRONT') and
                                 (s['is_key_frame'])]

    for token in tqdm.tqdm(sample_data_camera_tokens):

        img_path = nusc.get_sample_data_path(token)
        print(f"Processing image: {img_path}")
        img = read_image(img_path, format="BGR")
        predictions = demo(img)

        # save the predictions in a text file
        if len(predictions["instances"]) > 0:
            pred_classes = predictions["instances"].pred_classes.cpu().numpy()
            pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
            
            # Usa o nome base da imagem para o txt
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_filename = os.path.join(args.output_txt, base_name + ".txt")
            with open(txt_filename, "w") as f:
                for cls, box in zip(pred_classes, pred_boxes):
                    # Salva categoria e coordenadas no formato solicitado
                    line = f"{int(cls)}, {box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}\n"
                    f.write(line)
        
        break
    
if __name__ == "__main__":
   main()