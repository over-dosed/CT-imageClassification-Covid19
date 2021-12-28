import numpy as np
import cv2
import config
from DNN_functions import *

imgPathNi = './dataset/miniCTscans/NiCT/NiCT3999.jpg'
labelNi = 'NiCT3999.jpg'
imgPathN = './dataset/miniCTscans/nCT/nCT5862.jpg'
labelN = 'nCT5862.jpg'
imgPathP = './dataset/miniCTscans/pCT/pCT1934.jpg'
labelP = 'pCT1934.jpg'

imgNi = np.asarray(cv2.imread(imgPathNi)[:,:,::-1])
imgN = np.asarray(cv2.imread(imgPathN)[:,:,::-1])
imgP = np.asarray(cv2.imread(imgPathP)[:,:,::-1])

Ni = imgNi.reshape(-1,1)
P = imgP.reshape(-1,1)
N = imgN.reshape(-1,1)

#load params
parameters = np.load('./model/DNN3000-1.npy', allow_pickle=True).item()

AL1,caches1 = L_model_forward(Ni, parameters)                         #forward
AL2,caches2 = L_model_forward(N, parameters)                         #forward
AL3,caches3 = L_model_forward(P, parameters)                         #forward

# get predict dic from config
new_dict = {v : k for k, v in config.classes.items()}

# get predict str
result1 = ''+labelNi+" Predict: "+new_dict[int(np.argmax(AL1, axis=0))] + ":"+'%.2f' % ((AL1.max())*100)+"%"+".png"
result2 = ''+labelN+" Predict: "+new_dict[int(np.argmax(AL2, axis=0))] + ":"+'%.2f' % ((AL2.max())*100)+"%"+".png"
result3 = ''+labelP+" Predict: "+new_dict[int(np.argmax(AL3, axis=0))] + ":"+'%.2f' % ((AL3.max())*100)+"%"+".png"

cv2.imshow(result1,imgNi)
cv2.imwrite(result1,imgNi)
cv2.imwrite(result2,imgN)
cv2.imwrite(result3,imgP)
