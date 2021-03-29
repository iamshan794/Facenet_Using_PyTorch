import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
import torchvision.utils
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import sklearn
from sklearn import svm
import pickle
import cv2


#Importing the pretrained vggface2 model

resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()

#Defining transform
trans=transforms.Compose([transforms.Resize((120,120)),transforms.ToTensor(),transforms.CenterCrop((100,100))])

#Importing the trained model
os.chdir('facenet_pytorch/models')

clf=pickle.load(open('custom_faces_6.sav', 'rb'))
os.chdir('..')
os.chdir('..')
cap=cv2.VideoCapture(0)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('result.mp4',fourcc, 20.0, (1920,1080))

cls_lab={-1:'Unidentified Person or no face',0:'Person_122',1:'shan',2:'Person_125',3:'Person_121',4:'Person_124',5:'Person_123'}
while True:
	ret,img_orig=cap.read()
	if(ret==False):
		print('VideoCapture not working or done')
		break
	#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(np.uint8(img_orig)).convert('RGB')
	img=trans(img)
	
	face_embed=resnet(torch.reshape(img,(1,3,100,100)).cuda()).detach().to('cpu').numpy()
	face_embed=face_embed[0]
	yout=clf.predict_proba(face_embed.reshape(1,512))
	probab=yout[0,np.argmax(yout)]
	res_cls=np.argmax(yout) if yout[0,np.argmax(yout)]>=.7 else -1
	user=cls_lab[res_cls]
	db='User : '+user+'  CScore:'+str(float(probab))
	cv2.putText(img_orig, db, (150,150),cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 1, cv2.LINE_AA)
	#out.write(img_orig)
	cv2.imshow('result',img_orig)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#out.release()
cap.release()
cv2.destroyAllWindows()