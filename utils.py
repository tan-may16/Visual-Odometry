import numpy as np
import cv2 
import scipy
import math 

def feature_extractor(img1, img2, kp1 = None, des1  = None, bf = None):
    
    orb = cv2.ORB_create(nfeatures=200,scaleFactor=1.2,nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=0, patchSize=31, fastThreshold=20)
    kp1, des1 = orb.detectAndCompute(img1,None)
    
    if kp1 is None and des1 is None:
        kp2, des2 = orb.detectAndCompute(img2,None)

    if bf is None:
    # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    return matches, kp1, kp2, des1, des2
    
    
def get_matched_features(matches,kp1,kp2,n_features):
    
    
    matched_kp1 = np.array([kp1[m.queryIdx] for m in matches])
    matched_kp2 = np.array([kp2[m.trainIdx] for m in matches])
    matched_kp1_pt = np.array([kp1[m.queryIdx].pt for m in matches])
    matched_kp2_pt = np.array([kp2[m.trainIdx].pt for m in matches])
    
    # Define number of features to be considered from sorted matches
    n_features = min(n_features, len(matches))
    
    # Slice the matched keypoints and keypoints points
    matched_kp1 = matched_kp1[:n_features]
    matched_kp2 = matched_kp2[:n_features]
    matched_kp1_pt = matched_kp1_pt[:n_features]
    matched_kp2_pt = matched_kp2_pt[:n_features]
    
    return matched_kp1, matched_kp2, matched_kp1_pt, matched_kp2_pt


def reprojection_error(params, pts_3d, pts_2d, K):
    
    R = params[:3]
    t = params[3:]

    # Project 3D points onto 2D plane using K, R, t
    pts_2d_projected, _ = cv2.projectPoints(pts_3d, R, t, K, None)
    error = np.linalg.norm(pts_2d - pts_2d_projected.squeeze(), axis=1)
    return np.sum(error ** 2)


def skew(theta):
    tx, ty, tz = theta
    t_hat = np.array([[0, -tz, ty],[tz, 0, -tx],[-ty, tx, 0]])
    return t_hat


def Rodrigues(r):
    theta = np.linalg.norm(r)
    r = r / theta
    R = np.eye(3) + np.sin(theta) * skew(r) + (1 - np.cos(theta)) * np.dot(skew(r), skew(r))
    return R

def invRodrigues(R):
    epsilon = 1e-8
    tolerance = 1 - epsilon

    skew_symmetric = (R - R.T) / 2
    vector_p = np.array([[skew_symmetric[2, 1]], [skew_symmetric[0, 2]], [skew_symmetric[1, 0]]])
    vector_norm = np.linalg.norm(vector_p)
    trace = np.trace(R)
    cosine = (trace - 1) / 2

    if np.isclose(vector_norm, 0) and np.isclose(cosine, 1, rtol=0, atol=tolerance):
        return np.zeros(3)
    
    if np.isclose(vector_norm, 0) and np.isclose(cosine, -1, rtol=0, atol=tolerance):
        
        for i in range(3):
            if not np.isclose(np.sum(R[:, i]), -1):
                vector_v = R[:, i]
                break
        vector_u = vector_v / np.linalg.norm(vector_v)
        angle = np.pi
        vector_r = vector_u * angle
        if np.sqrt(np.sum(vector_r**2)) == np.pi and ((vector_r[0, 0] == 0. and vector_r[1, 0] == 0. and vector_r[2, 0] < 0) or (vector_r[0, 0] == 0. and vector_r[1, 0] < 0) or (vector_r[0, 0] < 0)):
            vector_r = -vector_r
        return vector_r
    
    else:
        vector_u = vector_p / vector_norm
        angle = np.arctan2(vector_norm, cosine)
        vector_r = vector_u * angle
        return vector_r



def Residual_error(K1, M1, p1, K2, p2, x):
    
    P, r, t = x[:-6], x[-6:-3], x[-3:]
    R = Rodrigues(r)
    M2 = np.hstack((R, np.reshape(t, (3, 1))))
    P = np.reshape(P, (-1, 3))
    
    homogeneous_points = np.vstack((P.T, np.ones((1, P.shape[0]))))
    img1_pred = K1 @ M1 @ homogeneous_points
    img1_pred = (img1_pred[:2, :] / img1_pred[2, :]).T
    img2_pred = K2 @ M2 @ homogeneous_points
    img2_pred = (img2_pred[:2, :] / img2_pred[2, :]).T
    residuals = np.concatenate([(p1 - img1_pred).reshape([-1]), (p2 - img2_pred).reshape([-1])])
    return residuals


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    R2 = M2_init[:, :3]
    t2 = M2_init[:, 3]
    r2 = invRodrigues(R2)
    x = np.concatenate([P_init.flatten(), r2.flatten(), t2])
    
    def error(x):
        return np.sum(Residual_error(K1, M1, p1, K2, p2, x)**2)
    
    
    options = {'gtol': 10, 'maxiter': 10}
    x_new = scipy.optimize.minimize(error, x, options = options).x
    
    r2 = x_new[-6:-3]
    t2 = x_new[-3:].reshape(3, 1)
    R2 = Rodrigues(r2)
    return R2, t2



# Code to extract ground truth from txt file or any other file
def Read_gt_odom():
    pass


def calculate_gt_error(pt1, pt2):
    error = np.linalg.norm(pt1 - pt2)
    return error


if __name__ == "__main__":
    
    #Read original Images
    img1=cv2.imread('0000000000.png',cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread('0000000001.png',cv2.IMREAD_GRAYSCALE)
    img1_coloured=cv2.imread('0000000000.png')
    img2_coloured=cv2.imread('0000000001.png')
    
    
    #Find corespondences, key points, descriptors
    matches, kp1, kp2, des1, des2 =feature_extractor(img1,img2)

    
    print('Matching features:',len(matches))
    print('Features in query Image (kp1):',len(kp1))
    print('Features in train Image (kp2):',len(kp2))
    
    #Define number of features to be considered
    n_features=len(matches)
    # n_features=50
    
    
    #Print feature matching indices in query image and train image
    # for i in range(len(matches)):
    #     print(matches[i].trainIdx,matches[i].queryIdx)
    
    #Get matching keypoints at same indices
    matched_kp1,matched_kp2, matched_kp1_pt, matched_kp2_pt=get_matched_features(matches,kp1,kp2,n_features)
    
    #Visualize correspondences
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:n_features],None)
    # cv2.imshow("Correspondences",img3)
    
    #Example: coordinates of 1st matched feature in both images
    print('1st feature coordinates in query Image',matched_kp1[0].pt)
    print('1st feature coordinates in training Image',matched_kp2[0].pt)
    
    ## Viualize Matched correspondences
    for i in range(len(matched_kp1)):
        cv2.circle(img1_coloured,(int(matched_kp1[i].pt[0]),int(matched_kp1[i].pt[1])),2,(0,0,255),2)
        cv2.circle(img2_coloured,(int(matched_kp2[i].pt[0]),int(matched_kp2[i].pt[1])),2,(0,0,255),2)
        
        
    cv2.imshow('First Image',img1_coloured)
    cv2.imshow('Second Image',img2_coloured)
    
    k=cv2.waitKey(0)
    if (k==ord('q')):
        cv2.destroyAllWindows()
    
        
        
        