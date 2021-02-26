from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import cv2  
import numpy as np
import apriltag

def detect_apriltag(gray, image):
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    #results = detector.detect(img=gray,True, camera_params=[544.021136,542.307110,308.111905,261.603373], tag_size=0.044)
    results = detector.detect(img=gray)
    
    if len(results) > 0:
        print("[INFO] {} total AprilTags detected".format(len(results)))
    else:
        print("No AprilTag Detected")
        return

    image = np.array(image)
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
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print("[INFO] tag family: {}".format(tagFamily))
        M,e1,e2 =detector.detection_pose(r,[544.021136,542.307110,308.111905,261.603373])
        #w:QR code length
        w=4.4
        t=[M[0][3]*w,M[1][3]*w,M[2][3]*w]
        
        dist=(t[0]**2+t[1]**2+t[2]**2)**0.5
        
        print("[INFO] dist:",dist," tag pose:",t)

def doloop():
    global depth, rgb
    while True:
        # Get a fresh frame
        (depth,_), (rgb,_) = get_depth(), get_video()
        
        # Build a two panel color image
        d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
        da = np.hstack((d3,rgb))
        
        # detect apriltag
        data = da[::2,::2,::-1]
        #image = cv2.imread(data)
        gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        detect_apriltag(gray, data)

        # Simple Downsample
        cv2.imshow('both', np.array(data))
        if cv2.waitKey(1000)==27:
            break

	
        
doloop()
