import numpy as np
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
    def on_release(self, event):
        print('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

def add(x, logfile):
    with open(logfile,"a+") as outfile:
        outfile.write(x + "\n")

photolist = '/home/wynmew/data/downloads/danbooru2018/origAll'
log = 'faceboxes'

idx=[]
for line in open(osp.join(photolist)):
    idx.append((os.path.join(line.strip())))

for i in range(len(idx)):
    #i=0
    img = idx[i]
    print(i, img)
    ims = cv2.cvtColor(cv2.imread(img), cv2.COLOR_RGB2BGR)

    plt.imshow(ims)
    a = Annotate()
    plt.show()
    if a.x0 is not None:
        x1, y1, x2, y2, w, h =a.x0,a.y0, a.x1, a.y1,a.x1-a.x0,a.y1-a.y0
        line = img +' ' + str(int(x1)) + ' ' + str(int(y1)) + ' ' + str(int(x2)) + ' ' + str(int(y2))+ ' ' + str(int(w))  + ' '+ str(int(h))
        #print(img, int(x1), int(y1), int(x2), int(y2), int(w), int(h))
        add(line,log)
        print(line)
    else:
        print(img, 'pass')
