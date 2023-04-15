import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import time
from utils import *


def main():
    
    start = time.time()
    curr_pose = np.zeros((3, 1))
    curr_rot = np.eye(3)
    trajectory = []
    
    # Define Camera Intrinsics
    #Define if groud truth if available.
    
    
    is_gt = False
    K = np.array([[9.591977e+02, 0.000000e+00, 6.944383e+02] ,[0.000000e+00, 9.529324e+02, 2.416793e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]]) 
    
    image_path = "images/"
    image_list = sorted(os.listdir(image_path), key=lambda x: int(os.path.splitext(x)[0]))
    
    if is_gt: gt = Read_gt_odom()
    
    
    scale = 1.0
    orb = cv2.ORB_create(nfeatures=200,scaleFactor=1.2,nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    prev_kp = None
    prev_des = None
    
    # Number of frames
    n = 100
    
    for index in range(n):
        print(index)
        if index == 0: continue
        
        curr_img = cv2.imread(image_path+image_list[index],0)
        
        
        if prev_kp is None:
            prev_img = cv2.imread(image_path+image_list[index - 1],0)
            prev_kp, prev_des = orb.detectAndCompute(prev_img,None)
            
        curr_kp, curr_des = orb.detectAndCompute(curr_img,None)
        
        matches = bf.match(prev_des,curr_des)
        matches = sorted(matches, key = lambda x:x.distance)
        
        n_features = len(matches)
        
        matched_kp1,matched_kp2, matched_kp1_pt, matched_kp2_pt = get_matched_features(matches,prev_kp,curr_kp,n_features)
        
        E, mask = cv2.findEssentialMat(matched_kp2_pt, matched_kp1_pt, K, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, mask = cv2.recoverPose(E, matched_kp2_pt, matched_kp1_pt, K)

        
        if is_gt:
            gt_pose = [gt[index,0], gt[index, 2]]
            prev_gt_pose = [gt[index - 1,0], gt[index - 1, 2]]

            scale = calculate_gt_error(gt_pose, prev_gt_pose)

        
        curr_pose += curr_rot.dot(t)*scale
        curr_rot = R.dot(curr_rot)
        trajectory.append(curr_pose.copy())
        
        prev_des = curr_des
        prev_kp = curr_kp
        
        
        ##### Bundle Adjustment ######
        
        # # Attempt 1
        
        # P1 = np.concatenate([K @ np.eye(3), K @ np.array([0,0,0]).reshape(-1, 1)], axis=1)
        # C1 = np.concatenate([np.eye(3), np.array([0,0,0]).reshape(-1, 1)], axis=1)
        
        # P2 = np.concatenate([K @ R, K @ t.reshape(-1, 1)], axis=1)
        # C2 = np.concatenate([ R, t.reshape(-1, 1)], axis=1)
        
        # P = cv2.triangulatePoints(P1, P2, matched_kp1_pt.T, matched_kp2_pt.T)
        # P = np.vstack((P[:3,:] / P[3,:])).T
        
        # R_optimized,t_optimized = bundleAdjustment(K, C1, matched_kp1_pt, K, C2, matched_kp2_pt, P)
        
        
        # #Attempt 2
        
        # # params_initial = np.hstack((cv2.Rodrigues(R)[0].flatten(), t.flatten()))
        # # bounds = [(-np.pi, np.pi)] * 3 + [(-10, 10)] * 3

        # # options = {'gtol': 0.1, 'maxiter': 100}

        # # result = scipy.optimize.minimize(reprojection_error, params_initial, args=(P, matched_kp2_pt, K), options=options)
        # # R_optimized = cv2.Rodrigues(result.x[:3])[0]
        # # t_optimized = result.x[3:].reshape(3, 1)
        
        # curr_pose += curr_rot.dot(t_optimized)*scale
        # curr_rot = R_optimized.dot(curr_rot)
        # trajectory.append(curr_pose.copy())
        
        # prev_des = curr_des
        # prev_kp = curr_kp
        
        
        cv2.imshow("Img", curr_img)
        a = cv2.waitKey(1)
        if a ==27: break
        
        
    cv2.destroyAllWindows() 
     
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:,0],trajectory[:,2])
    if is_gt: plt.plot(gt[:n,0],gt[:n,2])
    end = time.time()
    print("Time:", end- start)
    plt.savefig("Odom.png")
    plt.show()
    
if __name__ == "__main__":
    main()