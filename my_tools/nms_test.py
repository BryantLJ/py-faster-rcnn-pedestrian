import matplotlib.pyplot as plt
import cPickle
import numpy as np
import cv2

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def vis_detections(im,dets, thresh=0.3):
    """Visual debugging of detections."""
    #######
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    plt.show()
    plt.imshow(im)
    #######
    for i in xrange(np.minimum( 1000,dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            #plt.cla()
            #plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=1)
                )
            #plt.title('{}  {:.3f}'.format(class_name, score))

NMS_THRESH = 0.3

with open('nms_test2.txt','rb') as fid:
    dets1=cPickle.load(fid)
im = cv2.imread('/home/bryant/py-faster-rcnn/data/demo/000043.jpg')

dets_pre_nms = dets1[0]
dets_post_nms = dets1[1]

vis_detections(im,dets_pre_nms)
vis_detections(im,dets_post_nms)


keep = py_cpu_nms(dets_pre_nms,NMS_THRESH)
dets = dets_pre_nms[keep,:]

exit()
