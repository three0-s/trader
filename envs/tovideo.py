import cv2 
from glob import glob
import os 

pathIn= '/root/won/render/6/230514_ETH_processed/1684075483/*.png'
pathOut = '/root/won/230514eth.mp4'
fps = 5
frame_array = []
paths = (glob(pathIn))
paths=sorted(paths)
for idx , path in enumerate(paths) : 
    if (idx % 2 == 0) | (idx % 5 == 0) :
        continue
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()