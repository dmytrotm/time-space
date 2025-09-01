import cv2
import numpy as np 
import os
import math
from typing import Protocol

def image_iterator(input_dir:str):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            yield (filename, cv2.imread(os.path.join(path,filename)))


class TapeFinderInterface(Protocol):
    def find_tape(self,img:cv2.UMat,debug:bool=False)-> (int,int,int,int):
        ...
    def find_and_draw_tape(self,img:cv2.UMat,debug:bool=False):
        x,y,w,h = self.find_tape(img,debug)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  
        return img

class BlackTapeFinder(TapeFinderInterface):

    def __init__(self,lightIterations:int=0,wiresIterations:int=2):
        self.lightIterations = lightIterations
        self.wiresIterations = wiresIterations

    def find_tape(self,img:cv2.UMat,debug:bool=False)-> (int,int,int,int):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray,(3,3))
        _, threshGray = cv2.threshold(blur,40,255,cv2.THRESH_BINARY)
    

        kernel = np.ones((5,5))
        eroded0 = cv2.erode(threshGray,kernel,iterations=self.lightIterations)
        dilated0 = cv2.dilate(eroded0,kernel,iterations=self.lightIterations)
        kernel = np.ones((3,3))
        dilated = cv2.dilate(dilated0,kernel,iterations=self.wiresIterations)
        eroded = cv2.erode(dilated,kernel,iterations=self.wiresIterations)
        contours, hierarchy= cv2.findContours(eroded,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        correct = []
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area< 100:  
                continue
        
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= img.shape[0]*0.5  or h >= img.shape[1]*0.7 or w > h:
                continue
            areas.append(area)
            correct.append((x, y, w, h))

        if debug: 
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            cv2.imshow('h',h)
            cv2.imshow('s',s)
            cv2.imshow('v',v)
            cv2.imshow("gray",blur)
            cv2.imshow("tgray",threshGray)
            cv2.imshow("dilated",dilated)
            cv2.imshow("dilated0",dilated0)
            cv2.imshow("eroded",eroded)
            cv2.imshow("eroded0",eroded0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if len(correct) == 0:
            return -1,-1,-1,-1
        biggest = np.argmax(areas)
        return correct[biggest]

class BlueTapeFinder(TapeFinderInterface):

    def find_tape(self,img:cv2.UMat,debug:bool=False)-> (int,int,int,int):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 100, 50])   
        upper_blue = np.array([140, 255, 255])  

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((3,3))
        eroded = cv2.erode(mask,kernel,iterations=5)
        dilated = cv2.dilate(eroded,kernel,iterations=5)

        contours, hierarchy= cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        correct = []
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h < w*2 or x == 0 or x + w == img.shape[0]:
                continue
            areas.append(area)
            correct.append((x, y, w, h))
        if len(correct) == 0:
            return -1,-1,-1,-1
        biggest = np.argmax(areas)
        if debug:
            cv2.imshow("blue", mask)
            cv2.imshow("eroded", eroded)
            cv2.imshow("dilated", dilated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return correct[biggest]


def lookatthechannels(img):
    ch1,ch2,ch3 = cv2.split(img)
    cv2.imshow('1',ch1)
    cv2.imshow('2',ch2)
    cv2.imshow('',ch3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dark_pixels(img:cv2.UMat)->cv2.UMat:
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    s = hsv[:,:,2]
    _, threshold = cv2.threshold(s,100,255,cv2.THRESH_BINARY)
    threshold = cv2.dilate(threshold,np.ones((3,3)),iterations=2)
    threshold = cv2.erode(threshold,np.ones((3,3)),iterations=4)
    return threshold

if __name__ == "__main__":
    
    debug = True
    if debug:
        finder = BlackTapeFinder(3,9)
        img = cv2.imread(f"tapes/black_3/IMG_1689.jpg")
        cv2.imshow("asd",img)
        result = finder.find_and_draw_tape(img,debug)
        cv2.imshow("res",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        path = f"tapes/blue"
        output = f"output/detected_tape_blue"
        os.makedirs(output, exist_ok=True)
        finder = BlueTapeFinder()
        for filename, img in image_iterator(path):
            result = finder.find_and_draw_tape(img)
            cv2.imwrite(os.path.join(output,filename), result)
        
        
        for i in range(6):
            if(i == 0):
                finder = BlackTapeFinder(0,3)
            elif i==1:
                finder = BlackTapeFinder(0,6)
            else:
                finder = BlackTapeFinder(3,6)
            j = i+1
            path = f"tapes/black_{j}"
            output = f"output/detected_tape_black_{j}"
            os.makedirs(output, exist_ok=True)
            for filename, img in image_iterator(path):
                result = finder.find_and_draw_tape(img)
                cv2.imwrite(os.path.join(output,filename), result)