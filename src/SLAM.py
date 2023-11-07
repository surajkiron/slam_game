import numpy as np 
import cv2
from visual_odometry import PinholeCamera, VisualOdometry

# #import ros
# import rospy
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import Imu
# from std_msgs.msg import String
#from cv_bridge import CvBridge
import sys
import time


cam = PinholeCamera(320, 240, 92, 92, 160, 120)
vo = VisualOdometry(cam, '')
##-----------
#CHANGE THIS
data_dir = "/home/arpan/Projects/RobotPerception/vis_nav_player/Player_SuperPoint/VLAD/data/"

img_id=0

prev_draw_x,prev_draw_y=290,90
traj_points=[]
prev_dir = np.array([20,0,20]).T
VPRIndex=1

class SLAM:
	def __init__(self):
		self.target_locations=[]
		self.navigate = False
		self.target_traj=  []

	def reset(self,target_locations):
		vo.cur_R = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
		angle = -np.pi/4
		print("np cos,sin",np.cos(angle),np.sin(angle))
		vo.cur_R[0][0] = np.cos(angle)
		vo.cur_R[2][0] = -np.sin(angle)
		vo.cur_R[0][2] = np.sin(angle)
		vo.cur_R[2][2] = np.cos(angle)
		vo.cur_R[1][1] = 1

		vo.cur_t[0],vo.cur_t[1],vo.cur_t[2]=0,0,0
		#traj_points.clear()
		self.target_locations = target_locations
		self.navigate = True


#def Newframe_callback(data):
	def getOdometryFromOpticalFlow(self,data):
		img = (data)
		#print(img.shape)
		#print(len(data.data))
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		global img_id
		global prev_draw_x,prev_draw_y,traj_points,prev_dir,VPRIndex
		img_id+=1
		start_time = time.time()
		vo.update(img, img_id)
		delta_time = time.time() - start_time

		cur_t = vo.cur_t
		cur_R = vo.cur_R
		print("cur_R = ",cur_R)
		# _,pitch,_ = vo.getEulerAngles(cur_R)
		# R_2d = np.array([[np.cos(pitch),-np.sin(pitch)],[np.sin(pitch),np.cos(pitch)]])
		dir=np.array([20,0,20]).T

		if(cur_R is not None):
			dir = cur_R @ dir

		
		prev_dir = dir


		if(img_id > 15):
			x, y, z = cur_t[0], cur_t[1], cur_t[2]
		else:
			x, y, z = 0., 0., 0.
			
			return np.eye(3),np.zeros((3,))
		#print("x,y,z = ",x,y,z)
		print("x,z = ",x,z)
		draw_x, draw_y = int(x)+290, int(z)+290



		traj = np.zeros((600,600,3), dtype=np.uint8)

		#draw camera loc
		cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
		cv2.line(traj, (draw_x,draw_y), (draw_x+int(dir[0]),draw_y+int(dir[2])), (255,255,0), 10)

		if(draw_x==prev_draw_x and draw_y==prev_draw_y):

			pass
		else:
			if not self.navigate:
				traj_points.append([draw_x,draw_y])
			else:
				self.target_traj.append([draw_x,draw_y])


			prev_draw_x,prev_draw_y = draw_x,draw_y



		#draw trajectory
		for i in range(1,len(traj_points)):
			cv2.line(traj, (int(traj_points[i-1][0]),traj_points[i-1][1]), (traj_points[i][0],traj_points[i][1]), (255,0,0), 3) 
		
		for i in range(1,len(self.target_traj)):
			cv2.line(traj, (int(self.target_traj[i-1][0]),self.target_traj[i-1][1]), (self.target_traj[i][0],self.target_traj[i][1]), (204,255,255), 3) 

		for target in self.target_locations:
			cv2.circle(traj, (int(target[0]),int(target[1])), 1, (0,0,255), 5)

		cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
		text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
		cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

		#cv2.imshow('Road facing camera', img)
		cv2.imshow('Trajectory', traj)
		cv2.waitKey(1)
		return draw_x,draw_y
	