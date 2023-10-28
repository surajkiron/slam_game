import cv2
import numpy as np

class Extractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._K = np.array([[92,0,160],[0,92,120],[0,0,1]])
        self.prev_R = np.identity(3)
        self.prev_t = np.zeros((3,1))

        

    def get_keypts_and_desc(self, img):
        # kp, des = self.orb.detectAndCompute(img, None)

        # detection
        feats = cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        
        return kps, des
    
    def get_pose(self, img,K):
        kp, des = self.get_keypts_and_desc(img)
        if self.last is not None and len(des)>0:  # from the second iteration
            matches = self.bf.match(des, self.last['des'])
            ret = [] # is list[(tuple)(tuple)]
            for m in matches:
                kp1 = kp[m.queryIdx].pt
                kp2 = self.last['kps'][m.trainIdx].pt
                ret.append([kp1, kp2])
            if len(matches)>0:    
                # Extract keypoints as numpy arrays
                kp1 = np.array([match[0] for match in ret], dtype=np.float32)
                kp2 = np.array([match[1] for match in ret], dtype=np.float32)
                # Estimate the fundamental matrix
                fundamental_matrix, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC, 3.0, 0.99)

                # Estimate the essential matrix from the fundamental matrix and camera matrices
                # (You need to provide camera matrices K1 and K2, which can be identity matrices if you don't have calibration information)
                essential_matrix = np.dot(np.dot(np.transpose(K), fundamental_matrix), K)
                _, R, t, mask = cv2.recoverPose(essential_matrix, kp1, kp2)
                
                return R,t
        self.last = {'kps':kp,'des':des}
        R = self.prev_R
        t = self.prev_t
        self.prev_R = R
        self.prev_t = t
        return R,t