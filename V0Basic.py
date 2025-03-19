#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:52:32 2025

@author: adminnio
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2


class MonoVisualOdometry():
    def __init__(self, dir):
        self.K, self.P = self.loadCalib(datadir,'calib.txt')
       # self.gtPoses = self.loadPoses(datadir,'poses.txt')
        self.images = self.loadImages(datadir,'image_0') #Change Image dir to true directory name
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor= 1.1)
        index_params = dict(algorithm=0, trees=20)
        search_params = dict(checks=150)
        self.flann = cv2.FlannBasedMatcher(index_params = index_params, search_params = search_params)
        self.P0 = none
        self.P1 = none
        
    @staticmethod
    def loadCalib(self, datadir, filename):
        calib_path = f"{datadir}/{filename}"
        
        # Load the calibration file
        calibration = pd.read_csv(calib_path, delimiter=' ', header=None, index_col=0)
        
        # Extract and reshape P0 and P1
        self.P0 = np.array(calibration.loc['P0:']).astype(float).reshape((3, 4))
        self.P1 = np.array(calibration.loc['P1:']).astype(float).reshape((3, 4))
        
        print("P0 matrix:\n", self.P0)
        print("P1 matrix:\n", self.P1)
        
        return P0,P1
        
#    def loadPoses(self,datadir, filename):
#        poses = []
        
    @staticmethod
    def loadImages(datadir, filename):
        image_paths = [os.path.join(datadir, filename) for file in sorted(os.listdir(filename))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        
        
    def getTransformationmatrix():
    def matchFeatures():
    def getEssentialMatrix():
    def decomposeEssentialMatrix():
        