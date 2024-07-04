# camera_frame.py

from __future__ import print_function
from basicmotiondetector import BasicMotionDetector
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2
from face import Face

print(cv2.getBuildInformation())

# initialize the video stream and allow it to warm up
print("[INFO] starting camera...")
webcam = VideoStream(src=0).start()
time.sleep(2.0)

# initialize the motion detector and total number of frames read
camMotion = BasicMotionDetector()
total = 0

frame_names_camera1 = []

# Initialize the face detector with frame_names_camera1
face_detector = Face(frame_names_camera1)

try:
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream and resize it
        frame = webcam.read()
        if frame is None:
            print("[ERROR] Frame not captured from one of the cameras. Exiting...")
            break
        
        # Resize frames to 640x640 pixels
        frame = cv2.resize(frame, (640, 640))
        
        # convert the frame to grayscale, blur it, and update the motion detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        locs = camMotion.update(gray)
        
        # check if motion was detected
        if len(locs) > 0:
            # loop over the locations of motion and draw rectangles around them
            for loc in locs:
                (x, y, w, h) = cv2.boundingRect(loc)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        # increment the total number of frames read
        total += 1
        if total <= float("inf"):
            frame_name_camera1 = f"frame_camera1_{total}.jpg"
            frame_names_camera1.append(frame_name_camera1)
            
            
            # Process the frame immediately
            face_detector.process_frames([frame_name_camera1])
        
        # display the frame
        cv2.imshow("Frame", frame)
        
        # draw the timestamp on the frame and display it
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # check if a key was pressed
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Program interrupted by user")

finally:
    # Do a bit of cleanup
    print("[INFO] Cleaning up...")
    cv2.destroyAllWindows()
    webcam.stream.release()
    webcam.stop()

    # Print the frame names
    print("Frame names for Camera 1:", frame_names_camera1)