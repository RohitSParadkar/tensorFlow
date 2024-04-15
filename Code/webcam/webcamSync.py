# # import the opencv library 
# import cv2 
  
  
# # define a video capture object 
# vid = cv2.VideoCapture(1) 
  
# while(True): 
      
#     # Capture the video frame 
#     # by frame 
#     ret, frame = vid.read() 
  
#     # Display the resulting frame 
#     cv2.imshow('frame', frame) 
      
#     # the 'q' button is set as the 
#     # quitting button you may use any 
#     # desired button of your choice 
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break
  
# # After the loop release the cap object 
# vid.release() 
# # Destroy all the windows 
# cv2.destroyAllWindows() 

from threading import Thread
import numpy as np
import cv2
import time

class vStream:
    def __init__(self,src) -> None:
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update,args=())
        self.thread.daemon =True
        self.thread.start()

    def update(self):
        while True:
            _,self.frame = self.capture.read()
    
    def getFrame(self):
        return self.frame
    

cam1 = vStream(0)
cam2 = vStream(1)
while True:
    try:
        myFrame1 = cam1.getFrame()
        myFrame2 = cam2.getFrame()
        myFrame3 = np.hstack((myFrame1,myFrame2))
        cv2.imshow("CombineFrame",myFrame3)
        # cv2.moveWindow("CombineFrame",0,0)
        # cv2.imshow("webcam1",myFrame1)
        # cv2.imshow("webcam2",myFrame2)
    except:
        print("frame not available")
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cam1.capture.release()
        cam2.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break