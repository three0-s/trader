# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from time import time 
import os


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)
        self.strlogpath = os.path.join(log_dir, f"{str(int(time()))}.txt")
        self.strlogF = open(self.strlogpath)


    def LogAndPrint(self, txt):
        print(txt)
        self.strlogF.write(txt+"\n")


    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def image_summary(self, tag, images, step):
        """Log a list of PIL images."""

        for i, img in enumerate(images):
            tensorImg = ToTensor(img)
            # Create a Summary value
            self.writer.add_image('%s/%d' % (tag, i), tensorImg, step)


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        self.writer.add_histogram(tag=tag, values=values)
        self.writer.flush()

    def close(self):
        self.writer.close()
        self.strlogF.close()