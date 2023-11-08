import numpy as np 
import cv2
import superPoint_Glue
#from lightglue import ALIKED, DISK, SIFT, LightGlue, SuperPoint

from pathlib import Path
import argparse
# import cv2
import matplotlib.cm as cm
import torch
# import numpy as np
import sys
sys.path.append("./SuperGluePretrainedNetwork")
from models.matching import Matching
#from models.utils import frame2tensor, make_matching_plot_fast

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 3000

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 2000
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

device = 'cuda'
matching = Matching(config).eval().to(device)

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 3000

lk_params = dict(winSize  = (21, 21), 
				maxLevel = 8,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
	if(st is None):
		return None,None
	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2
	


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]
		


class VisualOdometry:
	def __init__(self, cam, annotations):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.cur_Normal= None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.prev_normal=None
		self.K = np.array([[cam.fx,0,cam.cx],[0,cam.fy,cam.cy],[0,0,1]])
		self.P = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
		self.frame_R=None
		self.frame_T=None
		# with open(annotations) as f:
		# 	self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = [0]*12#self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = [0]*12#self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	def getEulerAngles(self,R):
		yaw=np.arctan2(R[1,0],R[0,0])
		pitch=np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
		roll=np.arctan2(R[2,1],R[2,2])

		return [roll,pitch,yaw]

	def getRotationFromYaw(self,R,yaw):
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
	
	def recoverPose(self,H,q1,q2):
		def sum_z_cal_relative_scale(R, t):
			# Get the transformation matrix
			T = self._form_transf(R, t)
			# Make the projection matrix
			P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

			# Triangulate the 3D points
			hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
			# Also seen from cam 2
			hom_Q2 = np.matmul(T, hom_Q1)

			# Un-homogenize
			uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
			uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

			# Find the number of points there has positive z coordinate in both cameras
			sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
			sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

			# # Form point pairs and calculate the relative scale
			# relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
			# 							np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
			return sum_of_pos_z_Q1 + sum_of_pos_z_Q2,1#, relative_scale

		# Decompose the essential matrix
		retval, R, t, normal = cv2.decomposeHomographyMat(H, self.K)
		if(len(R)==1):
			R = np.array(R)
			t = np.array(t)
			normal = np.array(normal)
			R = np.reshape(R[0],(3,3))
			t = np.reshape(t[0],(3,1))
			normal = np.reshape(normal[0],(3,1))
			return R,t,normal
		
		else:
			#print("R = ",R," t= ",t)

		# Make a list of the different possible pair
		# Check which solution there is the right one
			z_sums = []
			relative_scales = []
			for i in range(len(R)):

				Ri,ti = R[i],t[i]
				Ri = np.array(Ri)
				ti = np.array(ti)
				Ri = np.reshape(Ri,(3,3))
				ti = np.reshape(ti,(3,1))
				z_sum, scale = sum_z_cal_relative_scale(Ri, ti)
				z_sums.append(z_sum)
				relative_scales.append(scale)

			# Select the pair there has the most points with positive z coordinate
			right_pair_idx = np.argmax(z_sums)
			right_pair = R[right_pair_idx],t[right_pair_idx],normal[right_pair_idx]
			relative_scale = relative_scales[right_pair_idx]
			R1, t,normal1 = right_pair
			#t = t * relative_scale
			R1 = np.array(R1)
			t = np.array(t)
			normal1 = np.array(normal1)
			R1 = np.reshape(R1,(3,3))
			t = np.reshape(t,(3,1))
			normal1 = np.reshape(normal1,(3,1))

			return R1, t, normal1


	def _form_transf(self,R, t):
		T = np.eye(4, dtype=np.float64)
		T[:3, :3] = R
		T[:3, 3] = np.reshape(t,(3,))
		return T

	
	def recoverPoseFromHomography(self,H):
		intrinsic_matrix = np.array([[self.focal, 0, self.pp[0]],
                             [0, self.focal,self.pp[1]],
                             [0, 0, 1]])

		retval, R, t, normal = cv2.decomposeHomographyMat(H, intrinsic_matrix)
		#print(R,t,normal)
		#if(self.prev_normal is None):
		#	self.prev_normal = np.array(normal[0])
		R = R[0]
		t = t[0]
		# else:	
		# 	for i in range(len(R)):
		# 		print(np.sum(self.prev_normal*np.array(normal[i])))
		# 		if(np.sum(self.prev_normal*np.array(normal[i]))>=0):
		# 			self.prev_normal = np.array(normal[i])
		# 			R = R[i]
		# 			t = t[i]
		# 			print("R,t,normal",R[i],t[i],normal[i])

		R = np.array(R)
		t = np.array(t)
		R = np.reshape(R,(3,3))
		t = np.reshape(t,(3,1))
		return R,t


	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		p1, p2, l, c, color, _ = superPoint_Glue.extract_keypoints_and_matches(matching, self.last_frame, self.new_frame)
		# l,c = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		# l,c, mkpts0, mkpts1, color = superPoint_Glue.extract_keypoints_and_matches(matching, self.last_frame, self.new_frame)
		temp_l = l
		l = c
		c = temp_l
		#superPoint_Glue.display_matching_results(self.last_frame, self.new_frame, p1, p2, l, c, color)
		if(l is None):
			return
		self.px_ref, self.px_cur = l,c
		error=[]
		for i in range(self.px_ref.shape[0]):
			error.append((self.px_ref[i][0]-self.px_cur[i][0])**2+(self.px_ref[i][1]-self.px_cur[i][1])**2)
		#print(sum(error))

		# H, mask = cv2.findHomography(self.px_cur, self.px_ref, method=cv2.RANSAC, confidence=0.995)
		# #self.cur_R,self.cur_t = self.recoverPoseFromHomography(H)
		# self.cur_R,self.cur_t,self.cur_Normal = self.recoverPose(H,self.px_ref, self.px_cur)
		self.cur_R = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
		angle = -np.pi/4
		self.cur_R[0][0] = np.cos(angle)
		self.cur_R[2][0] = -np.sin(angle)
		self.cur_R[0][2] = np.sin(angle)
		self.cur_R[2][2] = np.cos(angle)
		self.cur_R[1][1] = 1

		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		
		p1, p2, l, c, color, _ = superPoint_Glue.extract_keypoints_and_matches(matching, self.last_frame, self.new_frame)
		# l,c = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		# l,c, mkpts0, mkpts1, color = superPoint_Glue.extract_keypoints_and_matches(matching, self.last_frame, self.new_frame)
		temp_l = l
		l = c
		c = temp_l
		#superPoint_Glue.display_matching_results(self.last_frame, self.new_frame, p1, p2, l, c, color)
		#print(frame_id)
		if(l is None):
			return
		self.px_ref, self.px_cur = l,c
		error=[]
		for i in range(self.px_ref.shape[0]):
			error.append((self.px_ref[i][0]-self.px_cur[i][0])**2+(self.px_ref[i][1]-self.px_cur[i][1])**2)

		
		H, mask = cv2.findHomography(self.px_cur, self.px_ref, method=cv2.RANSAC, confidence=0.995)

		#R,t = self.recoverPoseFromHomography(H)
		R,t,self.cur_Normal = self.recoverPose(H,self.px_ref, self.px_cur)

		if(np.linalg.norm(t)>0.06):
			E, mask = cv2.findEssentialMat(self.px_cur, self.px_cur, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
			_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)

		
		self.frame_R=R
		self.frame_T=t
		absolute_scale = 2

		if(absolute_scale > 0.01):
			if(self.cur_t is None):
				self.cur_t = absolute_scale*(t) 
			# 	self.cur_R = R
			# else:
			self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
			self.cur_R = R. dot(self.cur_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
