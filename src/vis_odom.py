import numpy as np 
import cv2
import superpoint_glue
from pathlib import Path
import matplotlib.cm as cm
import sys
sys.path.append("./SuperGluePretrainedNetwork")
from models.matching import Matching

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
            'max_keypoints': -1
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
	
	def recoverPose(self, H, q1, q2):
		# A helper function to calculate the sum of Z coordinates that are positive for both cameras,
		# and also calculate the relative scale (unused here, but potentially useful for further development).
		def sum_z_cal_relative_scale(R, t):
			# Form the transformation matrix from rotation and translation
			T = self._form_transf(R, t)
			# Construct the projection matrix
			P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

			# Triangulate points in homogeneous coordinates
			hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
			# Transform the points as seen from the second camera
			hom_Q2 = np.matmul(T, hom_Q1)

			# Convert from homogeneous to Euclidean coordinates
			uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
			uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

			# Count the number of points with positive depth for both camera views
			sum_of_pos_z_Q1 = np.sum(uhom_Q1[2, :] > 0)
			sum_of_pos_z_Q2 = np.sum(uhom_Q2[2, :] > 0)

			return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, 1

		# Decompose the homography matrix to recover possible rotations and translations
		retval, R, t, normal = cv2.decomposeHomographyMat(H, self.K)
		
		# If only one solution exists, return it directly
		if(len(R) == 1):
			R = np.array(R)
			t = np.array(t)
			normal = np.array(normal)
			R = np.reshape(R[0], (3, 3))
			t = np.reshape(t[0], (3, 1))
			normal = np.reshape(normal[0], (3, 1))
			return R, t, normal
		
		else:
			# If multiple solutions exist, determine the correct one
			z_sums = []
			relative_scales = []
			
			# Evaluate each solution
			for i in range(len(R)):
				Ri, ti = R[i], t[i]
				z_sum, scale = sum_z_cal_relative_scale(Ri, ti)
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

			return R1, t, normal1



	def _form_transf(self,R, t):
		T = np.eye(4, dtype=np.float64)
		T[:3, :3] = R
		T[:3, 3] = np.reshape(t,(3,))
		return T

	
	def recoverPoseFromHomography(self, H):
		# Define the intrinsic camera matrix using the focal length and principal point
		intrinsic_matrix = np.array([[self.focal, 0, self.pp[0]],
									[0, self.focal, self.pp[1]],
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
		return R, t



	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		# Extract keypoints and matches between the last and the new frame using SuperPoint and SuperGlue
		p1, p2, l, c, color, _ = superpoint_glue.extract_keypoints_and_matches(matching, self.last_frame, self.new_frame)
		
		# Swap l and c to maintain consistency with the expected variable names for current and reference points
		temp_l = l
		l = c
		c = temp_l
		
		# Display the matching results visually
		superpoint_glue.display_matching_results(self.last_frame, self.new_frame, p1, p2, l, c, color)
		
		# If no points are left after matching, return without processing further
		if(l is None):
			return
		
		# Store the matched points as the current and reference points
		self.px_ref, self.px_cur = l, c
		
		# Calculate the error (distance) between matched points for potential use in filtering or evaluation
		error = []
		for i in range(self.px_ref.shape[0]):
			error.append((self.px_ref[i][0] - self.px_cur[i][0])**2 + (self.px_ref[i][1] - self.px_cur[i][1])**2)
		
		# Initialize the rotation matrix with a 45 degree rotation around the X-axis
		self.cur_R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
		angle = -np.pi/4  # 45 degrees
		self.cur_R[0][0] = np.cos(angle)
		self.cur_R[2][0] = -np.sin(angle)
		self.cur_R[0][2] = np.sin(angle)
		self.cur_R[2][2] = np.cos(angle)
		self.cur_R[1][1] = 1
		
		# Advance the frame stage to default since this is the second frame being processed
		self.frame_stage = STAGE_DEFAULT_FRAME
		
		# Update the reference points to the current points for tracking in the next iteration
		self.px_ref = self.px_cur


	def processFrame(self, frame_id):
		# Extract keypoints and matches between the last and the new frame
		p1, p2, l, c, color, _ = superpoint_glue.extract_keypoints_and_matches(matching, self.last_frame, self.new_frame)
		# Swap l and c to maintain consistency with the expected variable names for current and reference points
		temp_l = l
		l = c
		c = temp_l
		# Display the matching results visually
		superpoint_glue.display_matching_results(self.last_frame, self.new_frame, p1, p2, l, c, color)
		
		# If no points are left after matching, return without processing further
		if(l is None):
			return
		
		# Store the matched points as the current and reference points
		self.px_ref, self.px_cur = l, c
		
		# Calculate the error (distance) between matched points for potential use in filtering or evaluation
		error = []
		for i in range(self.px_ref.shape[0]):
			error.append((self.px_ref[i][0] - self.px_cur[i][0])**2 + (self.px_ref[i][1] - self.px_cur[i][1])**2)

		# Find the homography matrix between the current and reference points
		H, mask = cv2.findHomography(self.px_cur, self.px_ref, method=cv2.RANSAC, confidence=0.995)
		
		# Recover the pose from the homography matrix
		R, t, self.cur_Normal = self.recoverPose(H, self.px_ref, self.px_cur)

		# Check the magnitude of translation and compute the essential matrix if significant
		if(np.linalg.norm(t) > 0.06):
			E, mask = cv2.findEssentialMat(self.px_cur, self.px_cur, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
			_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)

		# Store the rotation and translation
		self.frame_R = R
		self.frame_T = t
		
		# Scale the translation to absolute scale
		absolute_scale = 2  # Assuming a scale factor of 2 for this example

		# If the absolute scale is significant, update the current translation and rotation
		if(absolute_scale > 0.01):
			if(self.cur_t is None):
				self.cur_t = absolute_scale * t
			else:
				self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
				self.cur_R = R.dot(self.cur_R)
		
		# If the number of reference points is below the threshold, detect new points
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		
		# Update the reference points for the next frame
		self.px_ref = self.px_cur

	# The update function handles the transition between different frame stages
	def update(self, img, frame_id):
		# Assert that the image is of the correct shape and is grayscale
		assert(img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
		# Update the new frame
		self.new_frame = img
		
		# Process the frame based on the current stage
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		
		# Update the last frame to the new frame for the next iteration
		self.last_frame = self.new_frame
