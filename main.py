import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import time
from utils import *
from realsense import *

def main(args):
    
    start = time.time()
    curr_pose = np.zeros((3, 1))
    curr_rot = np.eye(3)
    trajectory = []
    
    #Define if groud truth if available.
    is_gt = args.is_gt
    
    # Define Camera Intrinsics (Define your own function to read intrinsics in utils.py)
    K = get_intrinsic(args)
    
    image_path = args.data_path
    image_list = sorted(os.listdir(image_path), key=lambda x: int(os.path.splitext(x)[0]))
    
    if is_gt: gt = Read_gt_odom()
    scale = 1.0
    
    is_orb = (args.feature_extractor == 'orb')
    if (is_orb):
        orb = cv2.ORB_create(nfeatures=600,scaleFactor=1.2,nlevels=8, edgeThreshold=61, firstLevel=0, WTA_K=2, scoreType=0, patchSize=31, fastThreshold=10)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        sift = cv2.SIFT_create(contrastThreshold=0.022)
        flann = cv2.FlannBasedMatcher()
    
    prev_kp = None
    prev_des = None
    
    # Number of frames
    if is_gt: n = len(gt)
    else: n = len(image_list) 
    for index in range(n):
        print(index)
        if index == 0: continue
        
        curr_img = cv2.imread(image_path+image_list[index],0)
        
        
        if prev_kp is None:
            prev_img = cv2.imread(image_path+image_list[index - 1],0)
            if is_orb: prev_kp, prev_des = orb.detectAndCompute(prev_img,None)
            else: prev_kp, prev_des = sift.detectAndCompute(prev_img,None)
                
        if is_orb:
            curr_kp, curr_des = orb.detectAndCompute(curr_img,None)
            matches = bf.match(prev_des,curr_des)
            matches = sorted(matches, key = lambda x:x.distance)
            n_features = len(matches)
            matched_kp1,matched_kp2, matched_kp1_pt, matched_kp2_pt = get_matched_features(matches,prev_kp,curr_kp,n_features)
            
        else:
            curr_kp, curr_des = sift.detectAndCompute(curr_img,None)
            matches = flann.knnMatch(prev_des, curr_des, k=2)
            # good_matches = []
            matched_kp1_pt = []
            matched_kp2_pt = []

            for m, n_ in matches:
                if m.distance < 0.7 * n_.distance:
                    # good_matches.append(m)
                    matched_kp1_pt.append(prev_kp[m.queryIdx].pt)
                    matched_kp2_pt.append(curr_kp[m.trainIdx].pt)
            matched_kp1_pt = np.array(matched_kp1_pt)
            matched_kp2_pt = np.array(matched_kp2_pt)
            
        
        
        E, mask = cv2.findEssentialMat(matched_kp2_pt, matched_kp1_pt, K, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, mask = cv2.recoverPose(E, matched_kp2_pt, matched_kp1_pt, K)

        
        if is_gt:
            gt_pose = [gt[index,0], gt[index, 2]]
            prev_gt_pose = [gt[index - 1,0], gt[index - 1, 2]]
            scale = calculate_gt_error(np.array(gt_pose), np.array(prev_gt_pose))

        
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
    # print(trajectory.shape)
    np.savetxt('trajectory.txt', trajectory.squeeze(), delimiter=',', fmt='%4f')
    plt.plot(trajectory[:,0],trajectory[:,2])
    if is_gt: 
        plt.plot(gt[:index,0],gt[:index,2])
        plt.legend(['Odometry','Ground Truth'])
    else: plt.legend(['Odometry'])
    end = time.time()
    print("Time:", end- start)
    if is_orb: plt.savefig("Odom_orb.png")
    else: plt.savefig("Odom_sift.png")
    plt.show()
    
def live(args):
    
    start = time.time()
    curr_pose = np.zeros((3, 1))
    curr_rot = np.eye(3)
    trajectory = []
    
    K = get_intrinsic(args)
    
    
    d435i = RealSense()
    
    is_orb = (args.feature_extractor == 'orb')
    
    if is_orb:
        orb = cv2.ORB_create(nfeatures=200,scaleFactor=1.2,nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        sift = cv2.SIFT_create(contrastThreshold=0.022)
        flann = cv2.FlannBasedMatcher()
    
    prev_kp = None
    prev_des = None
    scale = 1.0
    
    i=0
    while True:
        ret, depth_frame, color_frame = d435i.get_frame()
        if i == 0:
            if is_orb: prev_kp, prev_des = orb.detectAndCompute(color_frame,None)
            else: sift.detectAndCompute(color_frame,None)
            
        if (i%10 == 0 and i != 0):    
            
            if is_orb:
                curr_kp, curr_des = orb.detectAndCompute(color_frame,None)
                
                matches = bf.match(prev_des,curr_des)
                matches = sorted(matches, key = lambda x:x.distance)
                
                n_features = len(matches)
                
                matched_kp1,matched_kp2, matched_kp1_pt, matched_kp2_pt = get_matched_features(matches,prev_kp,curr_kp,n_features)
            else:
                curr_kp, curr_des = sift.detectAndCompute(color_frame,None)
                matches = flann.knnMatch(prev_des, curr_des, k=2)
                # good_matches = []
                matched_kp1_pt = []
                matched_kp2_pt = []

                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        # good_matches.append(m)
                        matched_kp1_pt.append(prev_kp[m.queryIdx].pt)
                        matched_kp2_pt.append(curr_kp[m.trainIdx].pt)
                matched_kp1_pt = np.array(matched_kp1_pt)
                matched_kp2_pt = np.array(matched_kp2_pt)
                
            
            
            E, mask = cv2.findEssentialMat(matched_kp2_pt, matched_kp1_pt, K, cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, mask = cv2.recoverPose(E, matched_kp2_pt, matched_kp1_pt, K)
            
            curr_pose += curr_rot.dot(t)*scale
            curr_rot = R.dot(curr_rot)
            trajectory.append(curr_pose.copy())
            
            prev_des = curr_des
            prev_kp = curr_kp
            print(i)
        
        # cv2.imshow("depth frame", depth_frame)
        cv2.imshow("Color frame", color_frame)

        key = cv2.waitKey(100)
        i+= 1
        
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    
    trajectory = np.array(trajectory)
    np.savetxt('trajectory.txt', trajectory, delimiter=',', fmt='%4f')
    print(trajectory)
    plt.plot(trajectory[:,0],trajectory[:,2])
    plt.plot(trajectory[:,0],trajectory[:,1])
    plt.plot(trajectory[:,1],trajectory[:,2])
    plt.legend(['Odom xz', 'Odom xy', 'Odom yz'])
    end = time.time()
    print("Time:", end- start)
    if is_orb: plt.savefig("Odom_orb.png")
    else: plt.savefig("Odom_sift.png")
    plt.show()
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Input Arguments')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--data_path', type=str, default='images/')  
    parser.add_argument('--feature_extractor', type=str, default = 'sift')
    parser.add_argument('--is_gt', action='store_true')
    
    args = parser.parse_args()
    if args.live:
        live(args)
    else:
        main(args)
    