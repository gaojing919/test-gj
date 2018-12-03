import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
def set_scale(boxes):
    b_scale = np.zeros(60)
    b_large_size = np.zeros(len(boxes))
    boxes = np.array(boxes)
    #print(boxes)
    #xmin = np.array(boxes[0][:])
    #ymin = boxes[:][1]
    #xmax = boxes[:][2]
    #ymax = boxes[:][3]
    w = np.array(boxes[:,2]-boxes[:,0])
    h = np.array(boxes[:,3]-boxes[:,1])
    size = np.array([boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]])
    #print(w)
    #print(h)
    #print(size)
    #b_large_size = math.sqrt(size[0,:]*size[1,:])

    for i in range(size.shape[1]):
      b_large_size[i] = math.sqrt(size[0,i]*size[1,i])
    #print(b_large_size)

    #b_large_size = math.sqrt(size[:,1]*size[:,2])
    #print(b_large_size)
    model_count = 0
    model_index = []
    for b_size in b_large_size:
        x = len(bin(int(b_size)))-2
        if(abs(b_size-2**x)<abs(b_size-2**(x-1))):
          y = x
        else:
          y = x-1
        print(b_size)
        #print()
        index = (y-3)*10+5

        #print(index)
        #i = len(bin(int(b_size)))-4
        #index = i*10-1
        if(index<0):
            index = 5
        if(index>59):
            index = 55
        print(index)
        if(b_scale[index]==0):
            b_scale[index]=1
            model_count = model_count+1
            model_index.append(index)
            #print(index)
    for i in range(b_scale.size):
        d = 0
        for x in model_index:
            if(abs(i-x)<25):
                d = d+abs(i-x)
        for j in model_index:
            if(abs(i-j)<25):
                norm = multivariate_normal(mean=j, cov=1/math.sqrt(2*math.pi))
                if(d!=0):
                	b_scale[i] = b_scale[i]+abs(i-j)*norm.pdf(i)/d
                if(math.isnan(b_scale[i])):
                	print("nan")
                	print(d)
                	print(norm.pdf(i))
                if(b_scale[i]>1):
                	print("big1")
    plt.plot(range(60),b_scale)
    plt.show()
        #print(i)
        #print(len(i)-2)
    #b_large_size[:] = bin(b_large_size)
    #for b in boxes:
    #print(b_large_size)

    #print("gj_boxes")
    #print(boxes)
    print(b_scale)
    #sys.exit()
    return b_scale


def computScale(bottom):
	anchor_scale = np.zeros(3)
	pre_scale = bottom.reshape(60)
	#pre_scale = bottom[4].data.reshape(60)
	#print(pre_scale)
	#plt.plot(range(60),pre_scale)
	#plt.show()
	#moxing average smoothed
	#m = np.mean(pre_scale)
	for i in range(12):
	    m=np.mean(pre_scale[5*i:5*i+5])
	    pre_scale[5*i:5*i+5] = pre_scale[5*i:5*i+5]-m
	#pre_scale[:] = pre_scale[:] - m
	#print(pre_scale)
	#plt.plot(range(60),pre_scale)
	#plt.show()
	#1D NMS
	mark = np.zeros(60)
	while(0 in mark):
	    index = np.array(np.where(mark==0)[0])
	    print(index)
	    #index.reshape(index[0].size)
	    #print(index)
	    unhandle = pre_scale[index]
	    maxpostion = np.argmax(unhandle)
	    maxscale_index = index[maxpostion]
	    mark[maxscale_index] = 1
	    for i in range(maxscale_index-3,maxscale_index+4):
	    	if(i>=0 and i<60):
	    		if(mark[i]==0):
	    			mark[i]=-1
	    print(mark)
	    #print(maxpostion)
	    #print(index[maxpostion])
	    #print(pre_scale[maxpostion])
	    #print(index)
	    #print(pre_scale)
	    #print(unhandle)
	    #break
	max_index = np.array(np.where(mark==1)[0])
	max_scale = pre_scale[max_index]
	#print(len(max_scale))
	#print(max_index)
	#print(max_scale)
	while(0 in anchor_scale):
		anchor_index = np.array(np.where(anchor_scale==0)[0])
		s = np.argmax(max_scale)
		if(max_scale[s]==-1):
			break
		max_scale[s] = -1
		a_s = 8*2**int(max_index[s]/10)
		if(a_s not in anchor_scale):
			anchor_scale[anchor_index[0]] = a_s
	while(0 in anchor_scale):
		anchor_index = np.array(np.where(anchor_scale==0)[0])
		if(16 not in anchor_scale):
			anchor_scale[anchor_index[0]] = 16
		if(32 not in anchor_scale):
			anchor_scale[anchor_index[0]] = 32
		if(64 not in anchor_scale):
			anchor_scale[anchor_index[0]] = 64
	anchor_scale.sort()
	print(anchor_scale)
	return anchor_scale
		#print(anchor_scale)
		#break
	#plt.plot(max_index,max_scale)	
	#plt.show()


'''
bottom = np.array([[[[ 0.65733141]],

  [[ 0.67838174]],

  [[ 0.68303752]],

  [[ 0.7030139 ]],

  [[ 0.68562067]],

  [[ 0.71634358]],

  [[ 0.66860563]],

  [[ 0.66600966]],

  [[ 0.72647899]],

  [[ 0.67099673]],

  [[ 0.66341269]],

  [[ 0.74891996]],

  [[ 0.74055398]],

  [[ 0.62982571]],

  [[ 0.62180424]],

  [[ 0.71617812]],

  [[ 0.81529796]],

  [[ 0.64585495]],

  [[ 0.65519547]],

  [[ 0.79283071]],

  [[ 0.74875742]],

  [[ 0.74019372]],

  [[ 0.76771939]],

  [[ 0.67828321]],
  [[ 0.65471745]],

  [[ 0.789473  ]],

  [[ 0.61968046]],

  [[ 0.6794042 ]],

  [[ 0.79995966]],

  [[ 0.68056536]],

  [[ 0.6804418 ]],

  [[ 0.67283338]],

  [[ 0.82726848]],

  [[ 0.66390222]],

  [[ 0.69322342]],

  [[ 0.68184978]],

  [[ 0.715891  ]],

  [[ 0.66189831]],

  [[ 0.65926462]],

  [[ 0.75116408]],

  [[ 0.65569502]],

  [[ 0.69899648]],

  [[ 0.68482399]],

  [[ 0.80524057]],

  [[ 0.65862119]],

  [[ 0.75591201]],

  [[ 0.73284549]],

  [[ 0.6992889 ]],

  [[ 0.73035067]],

  [[ 0.65075529]],

  [[ 0.77715284]],

  [[ 0.74609548]],

  [[ 0.68534094]],

  [[ 0.66654861]],

  [[ 0.75571972]],

  [[ 0.73917478]],

  [[ 0.71154827]],

  [[ 0.706047  ]],

  [[ 0.7055971 ]],

  [[ 0.66542   ]]]])
computScale(bottom)
'''
#print(8*2**int(1/10))
boxes = [[  0 , 33 , 43,  85],[ 78 , 85 ,79, 90],[  4 , 69 ,200 ,124],[ 40 ,165 , 90, 210]]
set_scale(boxes)