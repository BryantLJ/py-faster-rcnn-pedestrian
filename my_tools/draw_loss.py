#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
with open('2016-11-29_16-59-09.log') as f:
    iter_num = []
    loss_value = []
    AUC = []
    for line in f.readlines():
        flag1 = line.find(', loss = ')
        flag2 = line.find('AUC is ')
        if flag1 != -1:
            # str[0]:Iteration **, str[1]: loss = **
            string = line.split(']')[1].split(',')
            cur1 = string[0].split(' ')[2]
            cur2 = string[1].split(' ')[3][:-1]
            iter_num.append(int(cur1))
            loss_value.append(float(cur2))
        if flag2 != -1:
            string = line.split(' ')
            cur = float(string[2])
            AUC.append(cur)
def get_odd_array(lista):
    length = len(lista)
    listb = []
    for i in range(length):
        if i%2 == 0:
            listb.append(lista[i])
    return listb

iter_num = get_odd_array(iter_num)
loss_value = get_odd_array(loss_value)

print len(iter_num),len(loss_value)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(iter_num,loss_value,'b',label='loss')
ax1.set_title('train_loss/auc - iteration_num',fontsize=14)
ax1.set_xlabel('iteration num',fontsize=14)
ax1.set_ylabel('loss',fontsize=14)
ax1.set_ylim(0,1.5)
ax1.grid(True)
ax1.legend()

A = 8*500
auc_iter_num = np.linspace(500,A,8)
ax2 = ax1.twinx()
ax2.plot(auc_iter_num,AUC,'r',linewidth = 2.0,label = 'auc')
ax2.set_ylabel('auc',fontsize=14)
ax2.legend(bbox_to_anchor=(1,0.915))

plt.show()        
