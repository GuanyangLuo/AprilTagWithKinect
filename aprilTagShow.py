import freenect
import cv2  
import numpy as np
import apriltag
import frame_convert2

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')

imgCount=0

def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])
    
def detect_apriltag(gray, image):
    
    global imgCount    
    
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    #results = detector.detect(img=gray,True, camera_params=[544.021136,542.307110,308.111905,261.603373], tag_size=0.044)
    results = detector.detect(img=gray)
    
    if len(results) > 0:
        print("[INFO] {} total AprilTags detected".format(len(results)))
    else:
        print("No AprilTag Detected")
        return image

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        #cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
       
        #print("[INFO] tag family: {}".format(tagFamily))
        M,e1,e2 =detector.detection_pose(r,[544.021136,542.307110,308.111905,261.603373])
        #w:QR code length
        w=4.4
        t=[M[0][3]*w,M[1][3]*w,M[2][3]*w]
        
        dist=(t[0]**2+t[1]**2+t[2]**2)**0.5
        
        showStr="dist:"+str(dist)
        
        cv2.putText(image, showStr, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        print("[INFO] dist:",dist," tag pose:",t)
    
    #save some photo
    if imgCount<50:
        cv2.imwrite("/home/gry/AprilDistImg/"+str(imgCount)+".jpg",image)   
        imgCount=imgCount+1
        print("image saved")
    
    return image


while 1:
    cv2.imshow('Depth', get_depth())
    
    #process the rgb image
    image=get_video()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image=detect_apriltag(gray,image.copy())
    
    cv2.imshow('Video', image)
    
    
    if cv2.waitKey(1) == 27:
        break

