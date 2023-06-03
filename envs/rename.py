import os 
from glob import glob 
import shutil

path = '/Users/yewon/Documents/traderWon/envs/eth/*.png'
todir='/Users/yewon/Documents/traderWon/envs/test'
files = glob(path)

for i, file in enumerate(files):
    name = f"{int(os.path.basename(file).replace('.png', '')):03d}"
    shutil.copy(file, os.path.join(todir, f'{name}.png'))