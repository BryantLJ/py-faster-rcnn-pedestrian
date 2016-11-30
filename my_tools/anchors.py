#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#     (xmin   ymin  xmax  ymax)
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.axis([-400,1000,-600,400])
for p in {
patches.Rectangle((-83,-39),100-(-83),56-(-39),fill=False,edgecolor='red'),
patches.Rectangle((-175,-87),192-(-175),104-(-87),fill=False,edgecolor='green'),
patches.Rectangle((-359,-183),376-(-359),200-(-183),fill=False,edgecolor='blue'),
patches.Rectangle((-55,-55),72-(-55),72-(-55),fill=False,edgecolor='red'),
patches.Rectangle((-119,-119),136-(-119),136-(-119),fill=False,edgecolor='green'),
patches.Rectangle((-247,-247),264-(-247),264-(-247),fill=False,edgecolor='blue'),
patches.Rectangle((-35,-79),52-(-35),96-(-79),fill=False,edgecolor='red'),
patches.Rectangle((-79,-167),96-(-79),184-(-167),fill=False,edgecolor='green'),
patches.Rectangle((-167,-343),184-(-167),360-(-343),fill=False,edgecolor='blue'),
patches.Rectangle((-0,-0),1000-(-0),-600-(-0),fill=False,hatch='.',edgecolor='black',linewidth=0.5),
}:
    ax.add_patch(p)
fig.savefig('anchor.jpg',dpi=150,bbox_inches='tight')
