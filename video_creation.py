# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:23:10 2022

@author: Luis-
"""

import cv2
import os

image_folder = 'plots/mollweide'
video_name = 'plots/mollweide-video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
