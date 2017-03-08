import numpy as np
import sys
sys.path.insert(0, '../Caffe/distribute/python')
import caffe
import cv2
import os

def initNet(root_path='./model',device_no=0):
    MODEL_FILE = root_path+'/deploy.prototxt'
    PRETRAINED = root_path+'/train_iter_40000.caffemodel'
    if device_no>=0:
        caffe.set_device(device_no)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
    return net

def processImage(net, path):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434]))
    transformer.set_raw_scale('data', 255)  # images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # channels in BGR order instead of RGB

    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            img=caffe.io.load_image(path + filename)
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

            # make sure you have created the output folder
            mapname= path + 'output/' + filename
            cv2.imwrite(mapname, map_final)

net = initNet('./model', -1)
path = './makoto/'
processImage(net, path)
