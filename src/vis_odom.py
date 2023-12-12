import numpy as np 
import cv2
import superpoint_glue
from pathlib import Path
import matplotlib.cm as cm
import sys
import torch
import lightWork


# sys.path.append("./SuperGluePretrainedNetwork")
# from models.matching import Matching

# sys.path.append("./LightGlue")
from lightglue import LightGlue, SuperPoint, DISK
import lightWork
# torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("medium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=1024, nms_radius=2).eval().cuda()  # load the extractor
matcher = LightGlue(features="superpoint").eval().cuda()
extractor.compile(mode="reduce-overhead")




# STAGE_FIRST_FRAME = 0
# STAGE_SECOND_FRAME = 1
# STAGE_DEFAULT_FRAME = 2
# kMinNumFeature = 3000




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

# device = 'cuda'
# matching = Matching(config).eval().to(device)

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 3000



def initialize_visual_odometry(cam_width, cam_height, cam_fx, cam_fy, cam_cx, cam_cy, cam_k1, cam_k2, cam_k3, cam_p1, cam_p2):
    vo_state = {
        'frame_stage': 0,
        'cam': {
            'width': cam_width,
            'height': cam_height,
            'fx': cam_fx,
            'fy': cam_fy,
            'cx': cam_cx,
            'cy': cam_cy,
            'k1': cam_k1,
            'k2': cam_k2,
            'k3': cam_k3,
            'p1': cam_p1,
            'p2': cam_p2
        },
        'new_frame': None,
        'last_frame': None,
        'cur_R': None,
        'cur_t': None,
        'cur_Normal': None,
        'px_ref': None,
        'px_cur': None,
        'focal': cam_fx,
        'pp': (cam_cx, cam_cy),
        'trueX': 0,
        'trueY': 0,
        'trueZ': 0,
        'detector': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
        'prev_normal': None,
        'K': np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]),
        'P': np.concatenate((np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]), np.zeros((3, 1))), axis=1),
        'frame_R': None,
        'frame_T': None
    }
    return vo_state



def getRotationFromYaw(R,yaw):
	R[0,0] = np.cos(yaw)
	R[0,1] = -np.sin(yaw)
	R[0,2] = 0
	R[1,0] = np.sin(yaw)
	R[1,1] = np.cos(yaw)
	R[1,2] = 0
	R[2,0] = 0
	R[2,1] = 0
	R[2,2] = 1
	return R

def _form_transf(R, t):
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	T[:3, 3] = np.reshape(t,(3,))
	return T

def sum_z_cal_relative_scale(vo_state,R, t,q1,q2):
	# Form the transformation matrix from rotation and translation
	T = _form_transf(R, t)
	# Construct the projection matrix
	P = np.matmul(np.concatenate((vo_state['K'], np.zeros((3, 1))), axis=1), T)

	# Triangulate points in homogeneous coordinates
	hom_Q1 = cv2.triangulatePoints(vo_state['P'], P, q1.T, q2.T)
	# Transform the points as seen from the second camera
	hom_Q2 = np.matmul(T, hom_Q1)

	# Convert from homogeneous to Euclidean coordinates
	uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
	uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

	# Count the number of points with positive depth for both camera views
	sum_of_pos_z_Q1 = np.sum(uhom_Q1[2, :] > 0)
	sum_of_pos_z_Q2 = np.sum(uhom_Q2[2, :] > 0)

	return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, 1

def recoverPose(vo_state, H, q1, q2):
	# A helper function to calculate the sum of Z coordinates that are positive for both cameras,
	# and also calculate the relative scale (unused here, but potentially useful for further development).


	# Decompose the homography matrix to recover possible rotations and translations
	retval, R, t, normal = cv2.decomposeHomographyMat(H, vo_state['K'])
	
	# If only one solution exists, return it directly
	if(len(R) == 1):
		R = np.array(R)
		t = np.array(t)
		normal = np.array(normal)
		R = np.reshape(R[0], (3, 3))
		t = np.reshape(t[0], (3, 1))
		normal = np.reshape(normal[0], (3, 1))
		return R, t, normal, vo_state
	
	else:
		# If multiple solutions exist, determine the correct one
		z_sums = []
		relative_scales = []
		
		# Evaluate each solution
		for i in range(len(R)):
			Ri, ti = R[i], t[i]
			z_sum, scale = sum_z_cal_relative_scale(vo_state,Ri, ti,q1,q2)
			z_sums.append(z_sum)
			relative_scales.append(scale)

		# Choose the solution with the maximum number of points with positive depth
		right_pair_idx = np.argmax(z_sums)
		right_pair = R[right_pair_idx], t[right_pair_idx], normal[right_pair_idx]
		R1, t, normal1 = right_pair

		# Reshape for consistency
		R1 = np.array(R1).reshape((3, 3))
		t = np.array(t).reshape((3, 1))
		normal1 = np.array(normal1).reshape((3, 1))

		return R1, t, normal1, vo_state


def recoverPoseFromHomography(vo_state, H):
	# Define the intrinsic camera matrix using the focal length and principal point
	intrinsic_matrix = np.array([[vo_state['focal'], 0, vo_state['pp'][0]],
								[0, vo_state['focal'], vo_state['pp'][1]],
								[0, 0, 1]])

	# Decompose the homography matrix to recover the possible rotations and translations
	retval, R, t, normal = cv2.decomposeHomographyMat(H, intrinsic_matrix)

	# Select the first solution from the possible solutions given by decomposition
	# The selection criteria or additional checks for the right solution can be implemented here
	R = R[0]
	t = t[0]

	# Reshape the rotation matrix R and translation vector t to be 3x3 and 3x1 respectively
	R = np.array(R).reshape((3, 3))
	t = np.array(t).reshape((3, 1))

	# Return the rotation matrix and translation vector
	return R, t, vo_state

def process_first_frame(vo_state):
    # Detect features in the new frame
    vo_state['px_ref'] = vo_state['detector'].detect(vo_state['new_frame'])
    # Convert detected features to a numpy array
    vo_state['px_ref'] = np.array([x.pt for x in vo_state['px_ref']], dtype=np.float32)
    # Update the frame stage
    vo_state['frame_stage'] = STAGE_SECOND_FRAME  # Replace STAGE_SECOND_FRAME with the actual value or constant

    # Return the updated state
    return vo_state


def process_second_frame(vo_state, STAGE_DEFAULT_FRAME):
    # Extract keypoints and matches between the last and the new frame using SuperPoint and SuperGlue
    # p1, p2, l, c, color, _ = superpoint_glue.extract_keypoints_and_matches(matching, vo_state['last_frame'], vo_state['new_frame'])
    l, c = lightWork.light(vo_state['last_frame'], vo_state['new_frame'])
    # Swap l and c to maintain consistency with the expected variable names for current and reference points
    l, c = c, l

    # Display the matching results visually
    # superpoint_glue.display_matching_results(vo_state['last_frame'], vo_state['new_frame'], p1, p2, l, c, color)
    
    # If no points are left after matching, return without processing further
    if l is None:
        return vo_state
    
    # Store the matched points as the current and reference points
    vo_state['px_ref'], vo_state['px_cur'] = l, c

    # Calculate the error (distance) between matched points for potential use in filtering or evaluation
    error = []
    for i in range(vo_state['px_ref'].shape[0]):
        error.append((vo_state['px_ref'][i][0] - vo_state['px_cur'][i][0])**2 + (vo_state['px_ref'][i][1] - vo_state['px_cur'][i][1])**2)

    # Initialize the rotation matrix with a 45 degree rotation around the X-axis
    angle = -np.pi / 4  # 45 degrees
    vo_state['cur_R'] = np.array([[np.cos(angle), 0, np.sin(angle)],
                                  [0, 1, 0],
                                  [-np.sin(angle), 0, np.cos(angle)]])

    # Advance the frame stage to default since this is the second frame being processed
    vo_state['frame_stage'] = STAGE_DEFAULT_FRAME

    # Update the reference points to the current points for tracking in the next iteration
    vo_state['px_ref'] = vo_state['px_cur']

    return vo_state


def process_frame(vo_state, kMinNumFeature, absolute_scale=2):
    # Extract keypoints and matches between the last and the new frame
    l, c = lightWork.light(vo_state['last_frame'], vo_state['new_frame'])
    # Swap l and c to maintain consistency with the expected variable names for current and reference points
    l, c = c, l

    # Display the matching results visually (if needed)
    # superpoint_glue.display_matching_results(vo_state['last_frame'], vo_state['new_frame'], p1, p2, l, c, color)
    
    # If no points are left after matching, return without processing further
    if l is None:
        return vo_state
    
    # Store the matched points as the current and reference points
    vo_state['px_ref'], vo_state['px_cur'] = l, c

    # Calculate the error (distance) between matched points
    error = [(vo_state['px_ref'][i][0] - vo_state['px_cur'][i][0])**2 + (vo_state['px_ref'][i][1] - vo_state['px_cur'][i][1])**2 for i in range(vo_state['px_ref'].shape[0])]

    # Find the homography matrix between the current and reference points
    H, mask = cv2.findHomography(vo_state['px_cur'], vo_state['px_ref'], method=cv2.RANSAC, confidence=0.995)
    
    # Recover the pose from the homography matrix
    R, t, vo_state['cur_Normal'],vo_state = recoverPose(vo_state, H, vo_state['px_ref'], vo_state['px_cur'])  # recover_pose needs to be defined or adjusted accordingly

    # Check the magnitude of translation and compute the essential matrix if significant
    if np.linalg.norm(t) > 0.06:
        E, mask = cv2.findEssentialMat(vo_state['px_cur'], vo_state['px_cur'], focal=vo_state['focal'], pp=vo_state['pp'], method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, vo_state['px_cur'], vo_state['px_ref'], focal=vo_state['focal'], pp=vo_state['pp'])

    # Store the rotation and translation
    vo_state['frame_R'] = R
    vo_state['frame_T'] = t

    # If the absolute scale is significant, update the current translation and rotation
    if absolute_scale > 0.01:
        if vo_state['cur_t'] is None:
            vo_state['cur_t'] = absolute_scale * t
        else:
            vo_state['cur_t'] = vo_state['cur_t'] + absolute_scale * vo_state['cur_R'].dot(t)
            vo_state['cur_R'] = R.dot(vo_state['cur_R'])

    # If the number of reference points is below the threshold, detect new points
    if vo_state['px_ref'].shape[0] < kMinNumFeature:
        vo_state['px_cur'] = vo_state['detector'].detect(vo_state['new_frame'])
        vo_state['px_cur'] = np.array([x.pt for x in vo_state['px_cur']], dtype=np.float32)
    
    # Update the reference points for the next frame
    vo_state['px_ref'] = vo_state['px_cur']

    return vo_state


def update(vo_state, img, kMinNumFeature, STAGE_DEFAULT_FRAME, STAGE_SECOND_FRAME, STAGE_FIRST_FRAME):
    # Assert that the image is of the correct shape and is grayscale
    assert(img.ndim == 2 and img.shape[0] == vo_state['cam']['height'] and img.shape[1] == vo_state['cam']['width']), "Frame: provided image has not the same size as the camera model or image is not grayscale"

    # Update the new frame
    vo_state['new_frame'] = img

    # Process the frame based on the current stage
    if vo_state['frame_stage'] == STAGE_DEFAULT_FRAME:
        vo_state = process_frame(vo_state, kMinNumFeature)
    elif vo_state['frame_stage'] == STAGE_SECOND_FRAME:
        vo_state = process_second_frame(vo_state, STAGE_DEFAULT_FRAME)
    elif vo_state['frame_stage'] == STAGE_FIRST_FRAME:
        vo_state = process_first_frame(vo_state)

    # Update the last frame to the new frame for the next iteration
    vo_state['last_frame'] = vo_state['new_frame']

    return vo_state
