import cv2
import numpy as np


def canny(img):

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # converting to gray scale image

    
    #cv2.imshow("mask_yw",mask_yw_image)
    gaussian = cv2.GaussianBlur(img,(5,5),0) # using gaussian blur to reduce noise while preserving edges

    canny = cv2.Canny(gaussian,50,150) # To detect the edges in the blurred image

    return canny

def get_coordinates(img,parameters):
    
    slope, intercept = parameters
    

    y1 = img.shape[0] # represents the height
    y2 = int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    imshape = img.shape
    vertices = np.array([[(419, 335), (530, 335), (width, height), (50, height)]], dtype=np.int32)
    mask = np.zeros_like(img) # creating a mask witht the same shape as of the image
    poly=cv2.fillPoly(mask,vertices,255)
    masked_image = cv2.bitwise_and(img,poly)
    return masked_image
    

cap = cv2.VideoCapture("solidWhiteRight.mp4")
while True:
    _, img = cap.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # converting to gray scale image

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)
    
    #cv2.imshow("mask_yw",mask_yw_image)
    gaussian = cv2.GaussianBlur(mask_yw_image,(5,5),0) # using gaussian blur to reduce noise while preserving edges

    canny = cv2.Canny(gaussian,50,150) # To detect the edges in the blurred image


    #cv2.imshow("canny",canny)

    mask = region_of_interest(canny)

    #cv2.imshow("mask",mask)
    lines = cv2.HoughLinesP(mask,2,np.pi/180,30,np.array([]),minLineLength=100,maxLineGap=180)

    left=[] # contains coordinates of respective average lines
    right=[] 
    for line in lines:
            x1,y1,x2,y2= line.reshape(4)
            parameters=np.polyfit((x1,x2),(y1,y2),1) # Fits a polynomial of degree 1 and returns a vector of the coeffeciants slope and y intercepts
            slope = parameters[0]
            intercept = parameters[1]
            if slope<0:
                left.append((slope,intercept))
            else:
                right.append((slope,intercept))
    
    print(left)
    left_average = np.average(left,axis=0)
    left_line = get_coordinates(img,left_average)
    right_average = np.average(right,axis=0)
    right_line = get_coordinates(img,right_average)
    avg_intercept = np.array([left_line,right_line])
    
    line_img = np.zeros_like(img) # creating an image of the same shape of the image
    if lines is not None:# checking if the array is empty
       for line in lines:
           x1,y1,x2,y2 = line.reshape(4)# converting a 2d array into a 1d array
           cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),10)


    detected=cv2.addWeighted(img,0.8,line_img,1,1)

    cv2.imshow("Final",detected)
    if(cv2.waitKey(2) == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()
