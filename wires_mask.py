import cv2
import numpy as np



def get_wires_mask(img: cv2.UMat):
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

    bordered_image = cv2.copyMakeBorder(
        eroded3,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=255 
    )
    contours, _ =  cv2.findContours(bordered_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    showimage = np.zeros_like(bordered_image)
    image_area = showimage.shape[0] * showimage.shape[1] * 0.5
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10000 or area >= image_area:
            continue
        cv2.drawContours(showimage,[contour],-1,255,-1)
    kernel = np.ones((11,11))
    dilated = cv2.dilate(showimage,kernel=kernel,iterations=5)
    eroded = cv2.erode(dilated,kernel=kernel,iterations=7) 
    return eroded[1:-1,1:-1]



if __name__ == "__main__":

    path = "rois/Taping incorrect (long fix)/IMG_1746.jpg"
    img = cv2.imread(path)
    mask = get_wires_mask(img)
    result = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()