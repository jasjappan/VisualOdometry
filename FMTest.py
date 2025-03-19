#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:00:25 2025

@author: adminnio
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
        calibpath = "/media/adminnio/Volume/Data/Datasets/Dataset/KITTI/dataset/sequences/01/calib.txt"
        calibration = pd.read_csv(calibpath, delimiter=' ', header=None, index_col=0)
        P0 = np.array(calibration.loc['P0:']).astype(float).reshape((3, 4))
        P1 = np.array(calibration.loc['P1:']).astype(float).reshape((3, 4))
        
        #print("P0 matrix:\n", P0)
        #print("P1 matrix:\n", P1)
        P = np.reshape(P0, (3, 4))
        K = P[0:3, 0:3]
        Pmat  = np.reshape(P1, (3, 4))
        Kmat = Pmat[0:3,0:3]
        img_leftPath = "/media/adminnio/Volume/Data/Datasets/Dataset/KITTI/dataset/sequences/01/image_0"
        img_rightPath = "/media/adminnio/Volume/Data/Datasets/Dataset/KITTI/dataset/sequences/01/image_1"
        
        image_paths_left = [os.path.join(img_leftPath, file) for file in sorted(os.listdir(img_leftPath))]
        image_paths_right = [os.path.join(img_leftPath, file) for file in sorted(os.listdir(img_rightPath))]
        # Read images in grayscale
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths_left]
        imagesR = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths_right]

        #plt.imshow(images[0],  cmap='gray')
        plt.axis("off")
        
        test_img = images[0]
        orb = cv2.ORB_create(nfeatures=5000,scaleFactor=1.5)
        queryKeypoints, queryDescriptors = orb.detectAndCompute(test_img,None)
        test_imgNext = images[1]
        test_img_right = imagesR[0]
        img2 = cv2.drawKeypoints(test_img,queryKeypoints,None, flags=0)
        
        queryKeypoints2, queryDescriptors2 = orb.detectAndCompute(test_imgNext,None)
        # finding nearest match with KNN algorithm
        index_params = dict(algorithm=0, trees=20)
        search_params = dict(checks=150)   # or pass empty dictionary
        #cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        # Initialize the FlannBasedMatcher
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
         
        matches = flann.knnMatch(queryDescriptors, queryDescriptors2, k=2)
        good = []
        try:
            for m, n in matches:
               if m.distance < 0.8 * n.distance:
                   good.append(m)
        except ValueError:
           pass

        draw_params = dict(matchColor = (0,155,0), # draw matches in green color
                singlePointColor = (155,0,0),
                matchesMask = None, # draw only inliers
                flags = 0)
        height, width = test_img.shape[:2]
        test_imgNext = cv2.resize(test_imgNext, (width, height))
        img3 = cv2.drawMatches(test_img, queryKeypoints, test_imgNext,queryKeypoints2, good ,None,**draw_params)
        #plt.imshow(img3)
        q1 = np.float32([queryKeypoints[m.queryIdx].pt for m in good])
        q2 = np.float32([queryKeypoints2[m.trainIdx].pt for m in good])
        E, _ = cv2.findEssentialMat(q1, q2, K, threshold=1)
        S = cv2.decomposeEssentialMat(E)
        print(S)
        if img3 is None:
            print("Error: Image not created. Check input descriptors.")
        else:
            cv2.imshow("in", img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
       


        
if __name__ == "__main__":
    main()