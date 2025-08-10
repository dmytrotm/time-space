import cv2
import numpy as np

def add_borders_and_extract_big_countours(img:cv2.UMat,how_big:int,border_color:np.uint8,border_size:int=1)->cv2.UMat:

        bordered_image = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color 
        )
        contours, _ =  cv2.findContours(bordered_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        showimage = np.zeros_like(bordered_image)
        image_area = showimage.shape[0] * showimage.shape[1] * 0.5
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < how_big or area >= image_area:
                continue
            cv2.drawContours(showimage,[contour],-1,255,-1)
        return showimage[border_size:-border_size,border_size:-border_size]


class WiresMaskExtractorInterface:
    def get_wires_mask(self,img: cv2.UMat)->cv2.UMat:
        ''' 
        Extract wires mask from the image
        '''
        pass

class HorizontalGaborFilterWiresMaskExtractor(WiresMaskExtractorInterface):
    def get_wires_mask(self,img: cv2.UMat)->cv2.UMat:
        ksize = 31  
        sigma = 4 
        theta = np.pi / 2 *0 
        lambd = 10.0  
        gamma = 0.5  
        psi = 0  
        blur= cv2.blur(img,(3,3))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create Gabor kernel
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)

        # Apply Gabor filter
        filtered_h = cv2.filter2D(hsv[:,:,2], cv2.CV_8UC3, gabor_kernel)

        _,binary_filtered_h = cv2.threshold(filtered_h,254,255,cv2.THRESH_BINARY)

        kernel = np.ones((5,5))

        eroded3 = cv2.erode(binary_filtered_h,kernel,iterations=4)
        image = add_borders_and_extract_big_countours(eroded3,10000,255)
        
        kernel = np.ones((11,11))
        dilated = cv2.dilate(image,kernel=kernel,iterations=5)
        eroded = cv2.erode(dilated,kernel=kernel,iterations=7) 
        return eroded

class BackgroundDifferenceWiresMaskExtractor(WiresMaskExtractorInterface):

    def __init__(self,backround_path:str):
        
        '''
        Needs a photo of the workspace withount wires 
        '''
        backround = cv2.imread(backround_path)
        self.backround = cv2.cvtColor(backround,cv2.COLOR_BGR2HSV)

    def get_wires_mask(self,img: cv2.UMat)->cv2.UMat:
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        diff = cv2.absdiff(self.backround[:,:,1],hsv[:,:,1])
        _,mask = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)
        

        image = add_borders_and_extract_big_countours(mask,10000,0)


        kernel = np.ones((11,11))
        image = cv2.dilate(image,kernel=kernel,iterations=2)
        image = cv2.erode(image,kernel=kernel,iterations=2)
        return image

class SaturationWiresMaskExtractor(WiresMaskExtractorInterface):
    '''
        currently the best method
    '''

    def get_wires_mask(self,img: cv2.UMat)->cv2.UMat:
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1]*2
        mask = cv2.adaptiveThreshold(s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,10)
        kernel = np.ones((3,3))
        mask = cv2.erode(mask,kernel,iterations=1)
        mask = cv2.dilate(mask,kernel,iterations=1)
        mask = add_borders_and_extract_big_countours(mask,15000,255)
        kernel = np.ones((11,11))
        mask = cv2.dilate(mask,kernel,iterations=2)
        kernel = np.ones((15,15))

        mask = cv2.erode(mask,kernel,iterations=2)
        return mask


if __name__ == "__main__":
    path = "rois/Taping incorrect (long fix)/IMG_1746.jpg"
    # extractor = HorizontalGaborFilterWiresMaskExtractor()
    #extractor = BackgroundDifferenceWiresMaskExtractor('rois/Taping incorrect (long fix)/IMG_1685.jpg')
    extractor = SaturationWiresMaskExtractor()
    img = cv2.imread(path)
    mask = extractor.get_wires_mask(img)
    result = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()