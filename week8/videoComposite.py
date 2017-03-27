import numpy as np
import cv2

def blend(frame1, frame2):
	h = min(len(frame1), len(frame2))
	w = min(len(frame1[0]), len(frame2[0]))
	mat = np.zeros((h, w, 3), np.uint8)
	for x in range(0, h):
		for y in range(0, w):
			mat[x][y] = frame1[x][y] * 0.5 + frame2[x][y] * 0.5
	return mat

video1 = cv2.VideoCapture('test2.mp4')
video1FrameCount = video1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
video1FrameNum = 0
video1CurrentFrame = None

# provide video
# video2 = cv2.VideoCapture('test2.mp4')

# webcam
video2 = cv2.VideoCapture(0)

video2FrameCount = video2.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
video2FrameNum = 0
video2CurrentFrame = None

while(video1.isOpened() and video2.isOpened()):
	if video1FrameNum == video1FrameCount:
		print("Video 1 finished")
		break
	# if video2FrameNum == video2FrameCount:
	# 	print("Video 2 finished")
	# 	break

	frameReady1, video1CurrentFrame = video1.read()
	frameReady2, video2CurrentFrame = video2.read()

	if frameReady1 and frameReady2:
		video2Resized = cv2.resize(video2CurrentFrame, (320, 320))
		result = blend(video1CurrentFrame, video2Resized)
		cv2.imshow('frame1', result)
		video1FrameNum = video1.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
		video2FrameNum = video2.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
		# process
	else:
		# video not ready, delay and try again
		cv2.waitKey(500)

	if cv2.waitKey(10) == 27:
		break

video1.release()
video2.release()
cv2.destroyAllWindows()
