from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import sys
sys.path.append("src/SuperGluePretrainedNetwork")
from models.matching import Matching
from models.utils import frame2tensor, make_matching_plot_fast
import glob
# from 

# config = {
#         'superpoint': {
#             'nms_radius': 4,
#             'keypoint_threshold': 0.005,
#             'max_keypoints': -1
#         },
#         'superglue': {
#             'weights': 'outdoor',
#             'sinkhorn_iterations': 20,
#             'match_threshold': 0.2,
#         }
#     }
device = 'cuda'
# matching = Matching(config).eval().to(device)

def load_matching_model(config):
    return Matching(config).eval()

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def extract_keypoints_and_matches(matching, image1, image2):
    # device = 'cuda'
    keys = ['keypoints', 'scores', 'descriptors']
    frame_tensor1 = frame2tensor(image1, device)
    frame_tensor2 = frame2tensor(image2, device)
    # device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    # matching = Matching(config).eval().to(device)
    
    data1 = matching.superpoint({'image': frame_tensor1})
    data1 = {k+'0': data1[k] for k in keys}
    data1['image0'] = frame_tensor1
    
    pred = matching({**data1, 'image1': frame_tensor2})
    
    kpts0 = data1['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].detach().cpu().numpy()

    
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    
    return kpts0, kpts1, mkpts0, mkpts1, color, matches

def estimate_essential_matrix_and_pose(mkpts0, mkpts1):

    mkpts0_homogeneous = np.column_stack((mkpts0, np.ones(len(mkpts0))))
    mkpts1_homogeneous = np.column_stack((mkpts1, np.ones(len(mkpts1))))


    mkpts0_2d = mkpts0_homogeneous[:, :2]
    mkpts1_2d = mkpts1_homogeneous[:, :2]
    
    essential_matrix, _ = cv2.findEssentialMat(mkpts0_2d, mkpts1_2d, focal=1.0, pp=(0, 0))
    _, R, t, _ = cv2.recoverPose(essential_matrix, mkpts0_2d, mkpts1_2d)
    
    return R, t, essential_matrix

def display_matching_results(image1, image2, kpts0, kpts1, mkpts0, mkpts1, color):
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    
    k_thresh = 0.005
    m_thresh = 0.2
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]
    
    out = make_matching_plot_fast(
        image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints='store_true', small_text=small_text)
    
    cv2.imshow('hi', out)
    cv2.waitKey(1)  # Display the image for 20 seconds
    # cv2.destroyAllWindows()

# image1 = load_image('/home/harshit/vis_nav_player/images/frame_0054.png')
# image2 = load_image('/home/harshit/vis_nav_player/images/frame_0056.png')
# kpts0, kpts1, mkpts0, mkpts1, color = extract_keypoints_and_matches(matching, image1, image2) 
# display_matching_results(image1, image2, mkpts0, mkpts1, mkpts0, mkpts1, color)

# images = glob.glob('/home/harshit/vis_nav_player/images/*.png')
# # print(images)
# imagesLen = len(images)
# for i in range(len(images)):
#     # print(i)
#     if i+1 == imagesLen:
#         break
#     else:
#         image1 = load_image(images[i])
#         image2 = load_image(images[i+1])
#         kpts0, kpts1, mkpts0, mkpts1, color = extract_keypoints_and_matches(matching, image1, image2)
#         # print(kpts0)
#         display_matching_results(image1, image2, mkpts0, mkpts1, mkpts0, mkpts1, color)

# print('done')