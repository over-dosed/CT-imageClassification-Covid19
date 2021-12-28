import numpy as np
import config
import cv2
import os
from PIL import Image

def Process():
    subfolders =  os.listdir('./weatherData')
    ftrain = open(config.trainPath,'a')
    fval = open(config.valPath,'a')
    for subfolder in subfolders:
        os.mkdir('./miniWeatherData/'+subfolder)
        images = os.listdir('./weatherData/'+subfolder)
        threshold = (len(images) * 4) / 5
        count = 0

        for image in images:
            strcontent = "" # the text ready to write in a line
            strcontent += ("./miniWeatherData/"+subfolder+"/"+image)
            count += 1

            img = Image.open("./weatherData/"+subfolder+"/"+image)
            out = img.resize((config.downSize, config.downSize),Image.ANTIALIAS) #resize image with high-quality
            out.save(strcontent,'png')

            if count > threshold:
                fval.write(strcontent+'_'+subfolder+'\n')
            else:
                ftrain.write(strcontent+'_'+subfolder+'\n')
        
        print('finish' + subfolder)


Process()