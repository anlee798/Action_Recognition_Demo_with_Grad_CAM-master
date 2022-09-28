import torch
import numpy as np
from PIL import Image
import time 
import cv2
import argparse
from models.model import generate_model
from utils import get_video_frame, show_cam_on_image
from grad_cam import GradCam
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models
import glob


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Action Recognition Demo")
    parser.add_argument("--video-path", type=str, default='./video_list/basketball.avi', help="Video path selection")
    parser.add_argument("--resume-path", type=str, default='./weights/ucf101_shufflevit2_1.0x_RGB_16_best.pth')
    return parser.parse_args()

#python demo.py --video-path video_list/basketball.avi --resume-path weights/ucf101_shufflenetv2hs_1.0x_RGB_16_best.pth
#python demo.py --video-path video_list/basketball.avi --resume-path weights/ucf101_ghostnet_1.0x_RGB_16_best.pth
if __name__ == '__main__':
    args = get_arguments()
    model= generate_model()
    checkpoint = torch.load(args.resume_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    stack_frame, end_frame = get_video_frame(args)
    stack_frame = stack_frame.cuda()
    pred5 = []

    #GHostNet
    #grad_cam = GradCam(model=model, feature_module=model.module.conv_head, target_layer_names=["2"])
    
    #ShuffleViT
    #grad_cam = GradCam(model=model, feature_module=model.module.last_conv2, target_layer_names=["2"])

    #ShuffleNetV2+HS
    grad_cam = GradCam(model=model, feature_module=model.module.conv_last, target_layer_names=["2"])
    #grad_cam = GradCam(model=model, feature_module=model.module.layer4, target_layer_names=["2"])
    if end_frame < 128:
         a = 1
    else:
         a = 2

    for frame in range(0, a):
        print("stack_frame",stack_frame.size())
        frame64 = stack_frame[:,:,frame*64:frame*64+16,:,:]
        print(frame64.size())
        start_time = time.time()
        pred = model(frame64)
        pred_time = time.time() - start_time
        mask = grad_cam(frame64, None)
        print("mask",mask.size())
        mask = mask.squeeze(0).squeeze(0)
        print("mask",mask.size())
        frame_cv = frame64.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        mask = mask.permute(1, 2, 0).cpu().numpy()        
        pred5 = np.array(torch.mean(pred, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
        show_cam_on_image(frame_cv, mask, frame, pred5, pred_time, args)
