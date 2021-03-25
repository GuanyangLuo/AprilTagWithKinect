import sys
import time
import copy
import numpy as np
import cv2
from openni import openni2, nite2, utils

openni2.initialize()
nite2.initialize()

devList = openni2.Device.open_all()
utList = []

try:
    #userTracker = nite2.UserTracker(dev)
    for i in range(0,len(devList)):
        utList.append(nite2.UserTracker(devList[i]))
        
except utils.NiteError as ne:
    logger.error("Unable to start the NiTE human tracker. Check "
                 "the error messages in the console. Model data "
                 "(s.dat, h.dat...) might be inaccessible.")
    sys.exit(-1)

csList=[]
for i in range(0,len(devList)):
    dev = devList[i]
    color_stream = dev.create_color_stream()
    color_stream.start()
    
    csList.append(color_stream)

while True:
    
    for i in range(0,len(utList)):
        userTracker = utList[i]
        
        frame = userTracker.read_frame()
        
        print("This is camera:",i)
        
        #skeleton tracking
        if frame.users:
            for user in frame.users:
                if user.is_new():
                    print("New human detected! Calibrating...")
                    userTracker.start_skeleton_tracking(user.id)
                elif user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED:
                    head = user.skeleton.joints[nite2.JointType.NITE_JOINT_HEAD]

                    confidence = head.positionConfidence
                    print("Head: (x:%dmm, y:%dmm, z:%dmm), confidence: %.2f" % (head.position.x,head.position.y,head.position.z,confidence))
    
        
        #RGB
        color_stream = csList[i]
        cframe = color_stream.read_frame()
        cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
        R = cframe_data[:, :, 0]
        G = cframe_data[:, :, 1]
        B = cframe_data[:, :, 2]
        cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
        # print(cframe_data.shape)
        cv2.imshow('color'+str(i), cframe_data)
        
        # q exit
        key = cv2.waitKey(1)
        if int(key) == 113:
            break
        
    

nite2.unload()
openni2.unload()