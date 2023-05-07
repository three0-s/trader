import cv2 
from glob import glob

pathIn= '/Users/yewon/Documents/traderWon/envs/test_render/7/162935/1683461084/*.png'
pathOut = '/Users/yewon/Documents/traderWon/envs/test_render/video.mp4'
fps = 3
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