import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg


class caltech(imdb):
    def __init__(self,train = 0,validation = 0,test = 0):
        self._name = 'caltech'
        self._image_path_root = '/home/bryant/MATLAB-tools/caltech/raw_image/'
        self._annotation_path = '/home/bryant/MATLAB-tools/caltech/raw_annotations/'
        if test == 1:
            self._image_set = ['set06/','set07/','set08/','set09/','set10/']
            self._image_stride = 30# sample 1 frame every 4 frames from _image_set
        elif validation == 1:
            self._image_set = ['set05/']
            self._image_stride = 30# sample 1 frame every 4 frames from _image_set  
        else:
            self._image_set = ['set00/','set01/','set02/','set03/','set04/']
            self._image_stride = 3# sample 1 frame every 4 frames from _image_set   
        self._classes = ['Background','person']
        self._class_num = len(self._classes)
        self._roidb_handler = self.gt_roidb
        self._image_index = self._load_image_index()
        if train==1:
            self._roidb = self.gt_roidb()    
    def _load_image_index(self):
        '''
        get image_index_list, ie. all image path list
        '''
        image_full_path_list = []
        count = 0
        for set_name in self._image_set:
            image_sets_path = os.listdir(self._image_path_root + set_name)
            image_sets_path.sort()
            for video_name in image_sets_path:
                set_video_sets = os.listdir(self._image_path_root + set_name + video_name)
                set_video_sets.sort()
                count = 0
                for im_name in set_video_sets:   
                    if count % self._image_stride == 0 :
                        image_full_path = self._image_path_root + set_name + video_name + '/' + im_name
                        image_full_path_list.append(image_full_path)
                    count = count + 1
        return image_full_path_list
    
    def image_path_at(self,i):
        '''
        return path of i-th image
        '''
        return self._image_index[i]
    
    def gt_roidb(self):
        '''
        '''  
        gt_roidb=[]
        new_img_list = []
        for index in self.image_index:
            roidb1,image_path = self._load_caltech_annotation(index)
            if roidb1:
                gt_roidb.append(roidb1)
                new_img_list.append(image_path)    
        self._image_index = new_img_list    
        return gt_roidb
    
    def _load_caltech_annotation(self,index):
        '''
        index: single image_path of image
        return roidb
        annotation txt format: 'person,x,y,w,h,occ_flag,vis_x,vis_y,visw,vis_h,**,**'
        '''
        pos1 = len(self._image_path_root)
        pos2 = index.find('.jpg')
        annotation_path = self._annotation_path + index[pos1:pos2] + '.txt'
        
        assert os.path.exists(annotation_path), 'Path does not exist: {}'.format(annotation_path)
        
        with open(annotation_path,'r') as f:
            lines = f.readlines()
            length = len(lines)
            count = 0
            if length>=2:
                for line_num in range(length-1):
                    strs = lines[line_num+1].split(' ')
                    if strs[0]== 'person' and strs[5] == '0'and strs[3]!='0' and strs[4]!='0':# not occluded, not far person
                        count = count + 1
        
#         if count==0:
#             return None,None
        
        boxes = np.zeros((count,4),dtype=np.uint16)
        gt_classes = np.zeros((count),dtype=np.int32)
        overlaps = np.zeros((count, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((count), dtype=np.float32)
        
        with open(annotation_path,'r') as f:
            lines = f.readlines()
            length = len(lines)
            count = 0
            if length>=2:
                for line_num in range(length-1):
                    strs = lines[line_num+1].split(' ')
                    if strs[0]== 'person' and strs[5] == '0'and strs[3]!='0' and strs[4]!='0':# not occluded, not far person
                        left = float(strs[1])
                        top = float(strs[2])
                        w = float(strs[3])
                        h = float(strs[4])
                        try:
                            assert(boxes[:, 2] >= boxes[:, 0]).all()
                            boxes[count,:] = [left,top,left+w,top+h]
                            gt_classes[count] = 1# person class
                            overlaps[count,1] = 1.0
                            seg_areas[count] = w * h
                            count = count + 1  
   
                        except:## there exists minus annotation!!! FATAL ERROR!!!
                            pass  
  
        overlaps = scipy.sparse.csr_matrix(overlaps)
    
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas},index
                
                
    def _load_caltech_evaluate_annotation(self,index):
        '''
        used to load annotation, including visable ratio info  
        annotation txt format: 'person,x,y,w,h,occ_flag,vis_x,vis_y,visw,vis_h,**,**'
        '''
        pos1 = len(self._image_path_root)
        pos2 = index.find('.jpg')
        annotation_path = self._annotation_path + index[pos1:pos2] + '.txt'
        
        assert os.path.exists(annotation_path), 'Path does not exist: {}'.format(annotation_path)
        
        with open(annotation_path,'r') as f:
            lines = f.readlines()
            length = len(lines)
            if length<=1:
                return None
        
        boxes = np.zeros((length-1,5),dtype=np.float32)

        with open(annotation_path,'r') as f:
            lines = f.readlines()
            length = len(lines)
            count = 0
            for line_num in range(length-1):
                strs = lines[line_num+1].split(' ')
                
                left = float(strs[1])
                top = float(strs[2])
                w = float(strs[3])
                h = float(strs[4])
                
                occ_flag = int(strs[5])
                vis_left = float(strs[6])
                vis_top = float(strs[7])
                vis_w = float(strs[8])
                vis_h = float(strs[9])    

                try:
                    assert(strs[0]=='person')
                    assert( w>0 and h>=50 and left>=5 and top>=5 and (left+w)<635 and (top+h)<475)
                    if occ_flag == 1:
                        assert (vis_left>=5 and vis_top>=5 and vis_w<=w and vis_h<=h)
                        assert (vis_w>=0 and vis_h>=0)
                        vis_ratio = 1.0*vis_w*vis_h/w/h
                        assert(vis_ratio>=0.65)
                    ig_flag = 0;
                except:
                    ig_flag = 1;
                                #aspect_ratio: w = 0.41*h
                center_x = left+w/2
                center_y = top+h/2
                w = 0.41*h
                left = center_x - w/2
                top = center_y - h/2    
                boxes[count,:] = [left,top,w,h,ig_flag]
                count = count + 1  
    
        return boxes
                
    def _write_det_results(self,img_path_list,all_dets,output_dir = 'output'):
        '''
        write detect result to txt format
        img_path_list :
                       'setID/seqID/frameID.jpg'
                       ...
        all_dets :
                       [x,y,w,h,score]
                       ...
        output txt:
                       setID,seqID,frameID,x,y,w,h,score
                       ...
        '''
        pass            
    
    def _compute_overlaps(self,box1,box2):
        '''
        compute ovelaps between box1 and box2
        box1:[x,y,w,h,score],det_box
        box2:[x,y,w,h,ig_flag], gt_box
        '''
        overlaps = 0
        w = min(box1[0]+box1[2],box2[0]+box2[2])-max(box1[0],box2[0])
        h = min(box1[1]+box1[3],box2[1]+box2[3])-max(box1[1],box2[1])
        inter = w*h
        ig_flag = box2[4]
        if w<=0 or h <=0:
            return overlaps
        
        if ig_flag == -1:
            overlaps = 1.0*inter/box1[2]/box1[3]
        else:
            overlaps = 1.0*inter/(box1[2]*box1[3]+box2[2]*box2[3]-inter)
            
        return np.float32(overlaps)
                
    def _do_python_eval(self, all_dets , output_dir = 'output'):
        '''
        evaluate the dets results, get Mean Average Precision(MAP)
        all_dets:
                [x,y,w,h,score]
        
        '''
        # step1: gtr loading & flitering, generate gtr[x,y,w,h,ig]
        gtr=[]
        for index in self.image_index:
            roidb1 = self._load_caltech_evaluate_annotation(index)
            gtr.append(roidb1)    
        
        
        # step2: dtr aspect_ratio & flitering
        for idx in range(len(all_dets)):
            img_dets = all_dets[idx]
            for idxj in range(len(img_dets)):
                det_boxes = img_dets[idxj]
                if det_boxes is not None:
                    left = det_boxes[0]
                    top = det_boxes[1]
                    w = det_boxes[2]-det_boxes[0]
                    h = det_boxes[3]-det_boxes[1]
                    score = det_boxes[4]
                    if h>=50/1.25:
                        center_x = left+w/2
                        center_y = top+h/2
                        w = 0.41*h
                        left = center_x - w/2
                        top = center_y - h/2
                        det_boxes = [left,top,w,h,score]
                    else:
                        det_boxes = np.zeros((1,5),dtype=np.float32)
                    img_dets[idxj] = det_boxes
            img_dets = img_dets.tolist()
            all_dets[idx] = filter(lambda x: np.any(x),img_dets)

        dtr = all_dets
        
        # step3:compute overlaps for i-th img  ; step4: match each gt_box and det_box
        num_img = len(gtr)
        OVERLAPS_THRESHOLD = 0.5

        for idx in range(num_img):
            det_boxes = dtr[idx]
            gt_boxes = gtr[idx]
            if gt_boxes is not None and det_boxes!=[]:
                det_num = len(det_boxes)
                gt_num = len(gt_boxes)
                det_boxes = sorted(det_boxes,key = lambda det_boxes:-det_boxes[4])
                gt_boxes[:,4] = -gt_boxes[:,4]
                gt_boxes = sorted(gt_boxes,key = lambda gt_boxes:-gt_boxes[4])
                det_boxes = np.hstack([det_boxes,np.zeros((det_num,1),dtype = np.float32)])
                
                overlaps_temp = np.zeros((det_num,gt_num),dtype=np.float32)
                for n in range(det_num):
                    for m in range(gt_num):
                        overlaps_temp[n][m] = self._compute_overlaps(det_boxes[n],gt_boxes[m]) 
                                    
                for idxj in range(det_num):
                    for idxg in range(gt_num):
                        if gt_boxes[idxg][4] == 0:# gt_box is not matched and not ignored 
                            if overlaps_temp[idxj][idxg]>=OVERLAPS_THRESHOLD:
                                det_boxes[idxj][5] = 1
                                gt_boxes[idxg][4] = 1
                                break
                            else:
                                continue
                        elif gt_boxes[idxg][4] == 1:# gt_box has already matched
                            continue
                        elif gt_boxes[idxg][4] == -1:# ignored gt_box can be matched for multi-times
                            if overlaps_temp[idxj][idxg]>=OVERLAPS_THRESHOLD:
                                det_boxes[idxj][5] = -1
                                break
                            else:
                                continue
                dtr[idx] = det_boxes
                gtr[idx] = gt_boxes
            elif det_boxes!=[]:# det_boxes is not [] ,gt_boxes is None
                det_num = len(det_boxes)
                det_boxes = np.array(det_boxes)
                det_boxes = np.hstack([det_boxes,np.zeros((det_num,1),dtype = np.float32)])
                dtr[idx] = det_boxes
            elif gt_boxes is not None:
                gt_boxes[:,4] = -gt_boxes[:,4]
                gtr[idx] = gt_boxes
            else:
                pass
            
        # step5: plot FPPI-MissRate Curve & compute AUC 
        fppi_ref = np.logspace(-2,0,9)
        dtr = filter(lambda x: x!=[],dtr)
        gtr = filter(lambda x: x is not None,gtr) 

        dtr = np.vstack(dtr[:])
        gtr = np.vstack(gtr[:])
        dtr = dtr[dtr[:,5]!=-1,:]
        gtr = gtr[gtr[:,4]!=-1,:]
             
        order = np.argsort(-dtr[:,4])
        tp = dtr[:,5]
        true_pos = tp[order]
        false_pos = np.ones((len(true_pos),))-true_pos
        total_pos = len(gtr)
        true_pos = np.cumsum(true_pos)
        false_pos = np.cumsum(false_pos)
        fppi = false_pos/num_img
        recall = true_pos/total_pos
        missRate = np.ones(recall.shape)-recall
        
        if 0:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(2e-4,50)
            ax.set_ylim(0.035,1e0)
            ax.grid(True)
            ymajorLocator = ticker.MultipleLocator(0.1)
            ax.yaxis.set_major_locator(ymajorLocator)
            yFormatter = ticker.FormatStrFormatter('%.1f')
            ax.yaxis.set_major_formatter(yFormatter)
            ax.plot(fppi,missRate)
            fig.show()
        
        missRate_ref = []
        for i in range(len(fppi_ref)):
            idx = np.where(fppi<fppi_ref[i])
            if len(idx[0]):
                missRate_ref.append(missRate[idx[0][-1]])
        AUC = np.exp(np.mean(np.log(missRate_ref)))
        #auc com
        print 'AUC is',AUC
        
if  __name__ == '__main__':
    '''
    '''
    d = caltech()
    res = d.roidb
    exit()
        
