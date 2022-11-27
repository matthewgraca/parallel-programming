"""
https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/#hello-world
"""
import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
#This tool allows you to convert neural network files (from various sources, like Tensorflow, Caffe or OpenVINO) to MyriadX blob file

"""
Allows us to see the frames from the color camera
"""
pipeline = depthai.Pipeline()
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

"""
We'll use the mobilenet-ssd network:
The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. 
This model is implemented using the Caffe* framework, so blob converter converts the nn files from Caffe to MyriadX blob file
"""
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
# Set path of the blob (NN model). We will use blobconverter to convert&download the model
# detection_nn.setBlobPath("/path/to/model.blob")
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

# connects color camera preview
cam_rgb.preview.link(detection_nn.input)

"""
Now we need both the color camera frames and the results of the nn's inference.
They are produced on the device, so they musy be transfered to the host machine.
This communication is handled by XLink. In this case from device->host, XLinkOut
"""
# send color camera frames to host
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# send nn inference results to host
xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

"""
Initialize depthai device with pipeline.
We also force usb2 communication because my thinkpad sucks
"""
with depthai.Device(pipeline, usb2Mode=True) as device:

    # host-side output queue to access results
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    
    # pull (consume) results from queue; set up placeholders that will hold it
    frame = None
    detections = []

    # nn bounding box coordinates are floats from <0,1> so they need to be normalized relative to pixel height/width
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    """
    consume results
    """
    while True:
        # fetch from queue
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        if in_nn is not None:
            detections = in_nn.detections

        # both results are 1D arrays; transform into useful data and display them
        if frame is not None:
            for detection in detections:
                # bounding box normalized to the rgb frame
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # draws rectangle on rgb frame w/ the location of the object detected 
                # (recall the object detected will be determined by mobilessd-net model, originally trained on tf, trained with PASCAL VOC0712)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)

        # exit infinite loop with 'q'
        if cv2.waitKey(1) == ord('q'):
            break
