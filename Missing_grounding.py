import cv2
import os
import numpy as np
import pandas as pd





def percentege_of_yellow(image: cv2.UMat,return_mask = False):
    '''
       Args:
       image: input image
       return_mask: if true renurn tuple, consist of mask of yellow pixels and percentege. else return just float 
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    total_pixels = mask.shape[0]*mask.shape[1]
    white_pixels = np.sum(mask)/255
    if return_mask:
        return (mask,white_pixels/total_pixels)
    else:
        return white_pixels/total_pixels

def is_grounding_missing(image: cv2.UMat, threshold: float = 0.001):
    p = percentege_of_yellow(img)
    return p < threshold




if __name__ == "__main__":
    folder = "rois/Missing grounding"

    right_image = "rois/Missing grounding/IMG_1746.jpg"
    wrong_image = "rois/Missing grounding/IMG_1685.jpg"

    exp = pd.read_csv("Expectation.csv")

    result = pd.DataFrame(columns=['Filename','%' ,'Result', 'Expected'])
    result.Filename = exp.Name
    result.Expected = exp["Missing grounding"]
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            name, _ = os.path.splitext(filename)
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            p = percentege_of_yellow(img)
            result.loc[result['Filename'] == filename, '%'] = p
    result.loc[result['%']<0.001,'Result'] = 1
    result.loc[result['%']>0.001,'Result'] = 0
    print(f"Baseline detector result: {sum(result['Result']==result['Expected'])}")