import numpy as np
import cv2
import caffe
import os

def blend(frame1, frame2, map):
	h = min(len(frame1), len(frame2))
	w = min(len(frame1[0]), len(frame2[0]))
	mat = np.zeros((h, w, 3), np.uint8)
	for x in range(0, h):
		for y in range(0, w):
			if map[x][y] > 0:
				mat[x][y] = frame2[x][y]
			else:
				mat[x][y] = frame1[x][y]
	return mat

def initNet(root_path='../saliency/model',device_no=0):
    MODEL_FILE = root_path+'/deploy.prototxt'
    PRETRAINED = root_path+'/train_iter_40000.caffemodel'
    if device_no>=0:
        caffe.set_device(device_no)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
    return net

def processImage(net, img):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434]))
    transformer.set_raw_scale('data', 255)  # images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # channels in BGR order instead of RGB

    #img=caffe.io.load_image(path)
    (H,W,C)=img.shape   #C=3

    #process the image
    imgData=transformer.preprocess('data',img)
    net.blobs['data'].data[...] = imgData
    net.forward()

    outmap=net.blobs['outmap'].data[0,0,:,:]
    map_final=cv2.resize(outmap,(W,H))
    map_final-=map_final.min()
    map_final/=map_final.max()
    map_final=np.ceil(map_final*255)

    return map_final

def processVideo():
	# background
	video1 = cv2.VideoCapture('test.mp4')
	video1FrameCount = video1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	video1FrameNum = 0
	video1CurrentFrame = None

	# foreground
	# video2 = cv2.VideoCapture('test2.mp4')
	video2 = cv2.VideoCapture(0)
	video2FrameCount = video2.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	video2FrameNum = 0
	video2CurrentFrame = None

	while(video1.isOpened() and video2.isOpened()):
		if video1FrameNum == video1FrameCount:
			print("Video 1 finished")
			break
# comment these out only when using webcam
#		if video2FrameNum == video2FrameCount:
#			print("Video 2 finished")
#			break

		frameReady1, video1CurrentFrame = video1.read()
		frameReady2, video2CurrentFrame = video2.read()

		if frameReady1 and frameReady2:
			video2Resized = cv2.resize(video2CurrentFrame, (320, 320))
			video1FrameNum = video1.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
			video2FrameNum = video2.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
			# process
			map = processImage(net, video2Resized)
			result = blend(video1CurrentFrame, video2Resized, map)
			cv2.imshow('result', result)
		else:
			# video not ready, delay and try again
			cv2.waitKey(500)

		if cv2.waitKey(10) == 27:
			break

	video1.release()
	video2.release()
	cv2.destroyAllWindows()

# GPU
# net = initNet('../saliency/model', 0)

# CPU
net = initNet('../saliency/model', -1)

processVideo()
