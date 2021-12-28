import numpy as np
import config
import cv2
import os
from PIL import Image
import random

def Process():
    subfolders =  os.listdir('./dataset/CTscans')
    ftrain = open(config.trainPath,'a')
    fval = open(config.valPath,'a')
    for subfolder in subfolders:
        os.mkdir('./dataset/miniCTscans/'+subfolder)
        images = os.listdir('./dataset/CTscans/'+subfolder)
        threshold = (len(images) * 9) / 10
        count = 0

        for image in images:
            strcontent = "" # the text ready to write in a line
            strcontent += ("./dataset/miniCTscans/"+subfolder+"/"+image)
            count += 1

            img = Image.open("./dataset/CTscans/"+subfolder+"/"+image)
            out = img.resize((config.downSize, config.downSize),Image.ANTIALIAS) #resize image with high-quality
            out.save(strcontent,'png')

            if count > threshold:
                fval.write(strcontent+'_'+subfolder+'\n')
            else:
                ftrain.write(strcontent+'_'+subfolder+'\n')
        
        print('finish_' + subfolder)

def get_line_offset(f):
    lines_start_offset=list()
    f.seek(0)
    lines_start_offset.append(f.tell())
    line = f.readline()
    while line:
        line=line.strip()
        lines_start_offset.append(f.tell())
        line = f.readline()
    return lines_start_offset   

def rewrite_file(f_in, f_out, lines_start_offset):
    for i in range(len(lines_start_offset)):
        f_in.seek(lines_start_offset[i], 0)
        line=f_in.readline()
        f_out.write(line)

Process()

if __name__ == "__main__":
    path1 = './val.txt'
    path2 = './val-1.txt'
    f = open(path1, 'r', encoding='utf-8')
    f_out = open(path2, 'w', encoding='utf-8')
    lines_start_offset = get_line_offset(f)
    random.shuffle(lines_start_offset)
    rewrite_file(f, f_out, lines_start_offset)
    f.close()
    f_out.close()