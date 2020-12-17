import cv2
import numpy as np


def canny(img):

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # converting to gray scale image

    gaussian = cv2.GaussianBlur(gray,(5,5),0) # using gaussian blur to reduce noise while preserving edges

    canny = cv2.Canny(gaussian,50,150) # To detect the edges in the blurred image

    return canny

def get_coordinates(img,parameters):
    try:
      slope, intercept = parameters
    except TypeError:
      slope, intercept = 0.001, 0

    y1 = img.shape[0] # represents the height
    y2 = int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    imshape = img.shape
    vertices = np.array([[(419, 340), (530, 340), (width, height), (50, height)]], dtype=np.int32)
    mask = np.zeros_like(img) # creating a mask with the same shape as of the image
    poly=cv2.fillPoly(mask,vertices,255)
    masked_image = cv2.bitwise_and(img,poly)
    return masked_image
    
def average(img,lines): #To optimise the lines
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
    
    left_average = np.average(left,axis=0)
    left_line = get_coordinates(img,left_average)

    
    right_average = np.average(right,axis=0)
    right_line = get_coordinates(img,right_average)
       
    return np.array([left_line,right_line])

def disp_lines(img,lines):
    line_img = np.zeros_like(img) # creating an image of the same shape of the image
    if lines is not None:# checking if the array is empty
       for line in lines:
           x1,y1,x2,y2 = line.reshape(4)# converting a 2d array into a 1d array
           cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),10) # Drawing all our lines onto the line image
    return line_img


img = cv2.imread("test_6.jpg")

lane_img = img # Making a copy of the image

canny=canny(img)

mask = region_of_interest(canny)

lines = cv2.HoughLinesP(mask,2,np.pi/180,30,np.array([]),minLineLength=100,maxLineGap=180)

avg_intercept = average(lane_img,lines)
disp_lines = disp_lines(lane_img,avg_intercept)


detected = cv2.addWeighted(lane_img,0.8,disp_lines,1,1)

cv2.imshow("Final",detected)
cv2.waitKey(0)
cv2.destroyAllWindows()
