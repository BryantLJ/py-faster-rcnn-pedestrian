#!/usr/bin/env python
import matplotlib.pyplot as plt
import cv2

im1 = cv2.imread('/home/bryant/py-faster-rcnn/data/demo/000003.jpg')
im2 = cv2.imread('/home/bryant/py-faster-rcnn/data/demo/000001.jpg')
bbox = [2,100,100,200]
score = 0.6
plt.show()
plt.imshow(im1)

plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
plt.imshow(im2)
exit()