import cv2 
from glob import glob

pathIn= '/mnt/won/render/prev/5/1926340/1683621573/*.png'
pathOut = '/mnt/won/render/prev/video_s2.mp4'
fps = 5
frame_array = []
paths = sorted(glob(pathIn))
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