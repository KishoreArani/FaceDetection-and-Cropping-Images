import cv2
import numpy as np
import os
from PIL import Image


cam = cv2.VideoCapture("/home/kishore/Desktop/facerecognition/videoplayback.mp4")

try: 
      
    
    if not os.path.exists('people'): 
        os.makedirs('people') 
  

except OSError: 
    print ('Error: Creating directory of data') 
  

currentframe = 0
count1 = 1
max_threshold = 110
min_threshold = 5


while(True): 
      
    
    ret,frame = cam.read() 
  
    if ret: 
         
        name = './people/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        
        if currentframe % 6 == 0:
            cv2.imwrite(name, frame) 
            count1 += 1
 
        currentframe += 1
    else: 
        break
  
cam.release() 
cv2.destroyAllWindows()


def dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]

    for i, person in enumerate(people):
            labels_dic[i] = person
            images.append(cv2.imread("people/" + person , cv2.IMREAD_UNCHANGED))
            labels.append(person)
       
    return (images, np.array(labels), labels_dic)

images, labels, labels_dic = dataset()



class FaceDetector(object):

    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 6
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        
        return faces_coord

def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = 0 
        x = x - 30
        y = y - 70
        h = h + 110
        w = w + 80
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
        
    return faces

def resize(images, size=(224, 224)):

    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm



def normalize_faces(image, faces_coord):

    faces = cut_faces(image, faces_coord)
    faces = resize(faces)
    
    return faces


count = 0
for image in images:
    
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    faces_coord = detector.detect(image, True)
    faces = normalize_faces(image ,faces_coord)
    for i, face in enumerate(faces):
        score = cv2.Laplacian(faces[i], cv2.CV_64F).var()
        if score < max_threshold and score > min_threshold:
            cv2.imwrite('%s.jpeg' % (count), faces[i])
            count += 1
        else:
            pass  
