import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')

while(cap.isOpened()):
	frameReady, frame = cap.read()
	if frameReady:
		cv2.imshow('frame', frame)
		frameNum = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
		print("frame " + str(frameNum))
		# process
	else:
		# delay and try again
		cv2.waitKey(500)
	
	if cv2.waitKey(10) == 27:
		break
	if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
		print("Video finished")
		break
	
cap.release()
cv2.destroyAllWindows()
