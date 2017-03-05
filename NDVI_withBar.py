#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import cv2
import time

#取得攝影機的影像
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

#colorBar的寬與高
bar_w = 20
bar_h = 256

#======製作colorBar======
bar=np.array([[]])
    #由上而下為255到0
for i in range(255,-1,-1):
    bar = np.append(bar,[i])

bar = np.reshape(bar,(bar_h,1))
b = np.reshape(bar,(bar_h,1))

for i in range(1,bar_w,1):
    bar = np.hstack((bar,b))

bar = np.uint8(bar)
#========================

#bar套用colorMap
bar = cv2.applyColorMap(bar, cv2.COLORMAP_JET)

#colorBar的xy位置，以block的左上角為原點
bar_x = 10
bar_y = 25

#白框的寬與高
block_w = bar_w +10 +30
block_h = bar_h +25 +10

#製作白框
block = np.empty((block_h,block_w,3),dtype=np.uint8)
block.fill(255)


#黑框的xy位置
blockB_x = bar_x -1
blockB_y = bar_y -1


#黑框的寬與高
blockB_w = bar_w +1 +1
blockB_h = bar_h +1 +1


#製作黑框
blockB = np.empty((blockB_h,blockB_w,3),dtype=np.uint8)
blockB.fill(0)


#白框裡放黑框
block[blockB_y:blockB_y+blockB_h,blockB_x:blockB_x+blockB_w] = blockB

#白框裡再放bar
block[bar_y:bar_y+bar_h,bar_x:bar_x+bar_w] = bar

#在白框標記NDVI的label
cv2.putText(block,'NDVI', (blockB_x, blockB_y-8), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(0, 0, 0), thickness=1)


#在白框標記數字(1 0 -1)
cv2.putText(block,'1', (blockB_x+bar_w+7, blockB_y+10), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(0, 0, 0), thickness=1)
cv2.putText(block,'0', (blockB_x+bar_w+7, blockB_y+int(blockB_h/2)+5), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(0, 0, 0), thickness=1)
cv2.putText(block,'-1',(blockB_x+bar_w+7, blockB_y+blockB_h-3), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(0, 0, 0), thickness=1)


#白框在frame的xy位置
block_x = frame.shape[1]- block_w -10
block_y = int(frame.shape[0]/2-block_h/2)


#捕捉滑鼠，按鍵存檔，檔名是時間
def onMouse_blue(event, x, y, flage, param):
    if event == cv2.EVENT_LBUTTONUP:
        cv2.imwrite("NDVI_blue-"+time.strftime("%Y%m%d-%H-%M-%S")+".png",NDVI_blue)


def onMouse_red(event, x, y, flage, param):
    if event == cv2.EVENT_LBUTTONUP:
        cv2.imwrite("NDVI_red-"+time.strftime("%Y%m%d-%H-%M-%S")+".png",NDVI_red)


cv2.namedWindow('NDVI_blue')
cv2.namedWindow('NDVI_red')

cv2.setMouseCallback('NDVI_blue', onMouse_blue)
cv2.setMouseCallback('NDVI_red', onMouse_red)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frameB = frame.copy()
    frameB = np.float32(frameB)
    B = frameB[:,:,0]
    G = frameB[:,:,1]
    R = frameB[:,:,2]

    #Measure NDVI===========
    #Blue Filter
    NDVI_blue = (R-B)/(R+B)
    
    #Red Filter
    NDVI_red = (B-R)/(R+B)
    #=======================
    NDVI_blue = np.uint8(NDVI_blue * (255/2) + (255/2))
    NDVI_red = np.uint8(NDVI_red * (255/2) + (255/2))


    #NDVI套用colorMap
    NDVI_blue = cv2.applyColorMap(NDVI_blue, cv2.COLORMAP_JET)
    NDVI_red = cv2.applyColorMap(NDVI_red, cv2.COLORMAP_JET)

    #在frame放入白框
    NDVI_blue[block_y:block_y+block_h,block_x:block_x+block_w] = block
    NDVI_red[block_y:block_y+block_h,block_x:block_x+block_w] = block


    # Display the resulting frame
    cv2.imshow('NDVI_blue',NDVI_blue)
    cv2.imshow('NDVI_red',NDVI_red)


    #按q離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


