import cv2, sys, numpy, os
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'


print ('Training...')

(images,lables,names, id) = ([],[],{},0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' +filename
            lable = id
            images.append(cv2.imread(path,0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112,92)

#create a numpy array from the two lists above
(images,lables) = [numpy.array(lis) for lis in [images,lables]]

#openCV trains a model from the iamges
model = cv2.createFisherFaceRecognizer()
model.train (images,lables)

#part2: use fisher reconigizer on camear stream
haar_cascade = cv2. CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)

# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    rval, im = webcam.read()
    im = cv2.flip(im,1,0)
    # mini = cv2.resize(im,(im.shape[1]/size, im.shape[0]/size))
    # faces = classifier.detectMultiScale(mini)

    gray = cv2.cvtColor (im,cv2.COLOR_BGR2GRAY)
    mini = cv2. resize(gray, (gray.shape[1]/size, gray.shape[0]/size))
    faces = haar_cascade.detectMultiScale(mini)

    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face= gray[y:y + h, x:x + w]
        face_resize = cv2. resize(face, (im_width, im_height))

        # try to recognize the face
        prediction = model.predict(face_resize)
        cv2. rectangle(im,(x,y),(x+w, y+h),(0,255,0),3)

        #write the name of recognized face

        if prediction[1]<500:
            cv2.putText(im,
                '%s - %.0f' % (names[prediction[0]],prediction[1]),
                    (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im,'Unknown',
            (x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))

    cv2.imshow("openCV",im)

    key = cv2. waitKey(10)
    if key == 27:
        break
