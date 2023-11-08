import numpy as np 
import cv2
from vis_odom import PinholeCamera, VisualOdometry
import time

# Initialize the camera parameters and visual odometry objects
cam = PinholeCamera(320, 240, 92, 92, 160, 120)
vo = VisualOdometry(cam, '')

# Frame index initialization
img_id = 0

# Variables for drawing the trajectory
prev_draw_x, prev_draw_y = 290, 90
traj_points = []
prev_direction = np.array([20, 0, 20]).T
VPRIndex = 1

class SLAM:
	def __init__(self):
		self.target_locations = []
		self.navigate = False
		self.target_traj = []

	def reset(self, target_locations):
		# Reset the current rotation and translation of the visual odometry to initial state
		vo.cur_R = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
		angle = -np.pi / 4
		vo.cur_R[0][0] = np.cos(angle)
		vo.cur_R[2][0] = -np.sin(angle)
		vo.cur_R[0][2] = np.sin(angle)
		vo.cur_R[2][2] = np.cos(angle)
		vo.cur_R[1][1] = 1

		# Reset current translation to origin
		vo.cur_t[0],vo.cur_t[1],vo.cur_t[2]=0,0,0
		# Clear the trajectory points
		self.target_locations = target_locations
		self.navigate = True

	def getOdometryFromOpticalFlow(self, data):
		# Convert the input data to an image format for processing
		img = data  # Assuming data is already in the correct format for update

		# Increment the global image ID to keep track of the number of processed frames
		global img_id, prev_draw_x, prev_draw_y, traj_points, prev_dir, VPRIndex
		img_id += 1

		# Start timing the update process for performance measurement
		start_time = time.time()
		# Update the visual odometry with the new image frame
		vo.update(img, img_id)
		# Measure how long the update took
		delta_time = time.time() - start_time

		# Retrieve the current estimated camera pose (position and rotation)
		cur_t = vo.cur_t
		cur_R = vo.cur_R

		# Initialize a direction vector for visualization purposes
		dir = np.array([20, 0, 20]).T
		if cur_R is not None:
			# Rotate the direction vector based on the current camera rotation
			dir = cur_R @ dir
		# Store the direction for future reference
		prev_dir = dir

		# If more than 15 frames have passed, use the estimated translation; otherwise, use zero
		if img_id > 15:
			x, y, z = cur_t[0], cur_t[1], cur_t[2]
		else:
			x, y, z = 0., 0., 0.
			# Return an identity matrix and a zero vector if not enough frames have passed
			return np.eye(3), np.zeros((3,))

		# Calculate the drawing coordinates for the current camera position on the trajectory image
		draw_x, draw_y = int(x) + 290, int(z) + 290

		# Initialize a blank image for drawing the trajectory
		traj = np.zeros((720, 720, 3), dtype=np.uint8)
		# Draw a circle at the current camera location on the trajectory image
		cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)

		# Define the scale for the direction arrow to represent camera orientation
		arrow_scale = 1
		# Calculate the end point of the arrow for visualization
		end_point_x = draw_x - int(dir[0] * arrow_scale)
		end_point_y = draw_y - int(dir[2] * arrow_scale)
		# Draw the arrowed line representing the camera's orientation
		cv2.arrowedLine(traj, (draw_x, draw_y), (end_point_x, end_point_y), (0, 0, 255), thickness=2)

		# Check if the current position is different from the previous one to update the trajectory
		if (draw_x, draw_y) != (prev_draw_x, prev_draw_y):
			if not self.navigate:
				traj_points.append([draw_x, draw_y])
			else:
				self.target_traj.append([draw_x, draw_y])
			# Update the previous position
			prev_draw_x, prev_draw_y = draw_x, draw_y

		# Draw the entire trajectory on the image
		for i in range(1, len(traj_points)):
			cv2.line(traj, (traj_points[i - 1][0], traj_points[i - 1][1]), (traj_points[i][0], traj_points[i][1]), (255, 125, 0), 2)

		# Draw the target trajectory if navigating towards targets
		for i in range(1, len(self.target_traj)):
			cv2.line(traj, (self.target_traj[i - 1][0], self.target_traj[i - 1][1]), (self.target_traj[i][0], self.target_traj[i][1]), (204, 255, 255), 2)

		# Display target locations on the trajectory image
		for target in self.target_locations:
			cv2.circle(traj, (int(target[0]), int(target[1])), 1, (0, 0, 255), 5)

		# Overlay text displaying the current coordinates onto the trajectory image
		text = "Coords: x=%2fm y=%2fm z=%2fm" % (x, y, z)
		cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

		# Show the trajectory image with the current camera location and trajectory
		cv2.imshow('Trajectory', traj)
		# Wait for 1 ms to allow for the image to be displayed properly
		cv2.waitKey(1)

		# Return the drawing coordinates for potential further use
		return draw_x, draw_y
