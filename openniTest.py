from openni import openni2
import numpy as np
import cv2

openni2.initialize()     # can also accept the path of the OpenNI redistribution

#dev = openni2.Device.open_any()
devList = openni2.Device.open_all()
print('dev:',devList)
dev = devList[0]
print(dev.get_device_info())

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
depth_stream.start()
color_stream.start()


while True:
    # Depth
    frame = depth_stream.read_frame()
    dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
    dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
    dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
    dpt2 *= 255
    dpt = dpt1 + dpt2
    cv2.imshow('dpt', dpt)

    # RGB
    cframe = color_stream.read_frame()
    cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
    R = cframe_data[:, :, 0]
    G = cframe_data[:, :, 1]
    B = cframe_data[:, :, 2]
    cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
    # print(cframe_data.shape)
    cv2.imshow('color', cframe_data)

    # q exit
    key = cv2.waitKey(1)
    if int(key) == 113:
        break

# 
depth_stream.stop()
color_stream.stop()
dev.close()
