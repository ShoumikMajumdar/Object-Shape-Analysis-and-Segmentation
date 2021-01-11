import cv2 as cv
import numpy as np
import os
import math

#Load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv.resize(img, (500, 500))
            images.append(img)
    return images


#Convert to grayscale and adaptinve thresholding
def preprocess(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #_, img = cv.threshold(img,80,255,cv.THRESH_BINARY)
    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return img


def main():
    folder = "CS585-BatImages/Gray"
    data = load_images_from_folder(folder)

    test_image = data[121]
    cv.imshow("object",test_image)
    threshold = preprocess(test_image)

    threshold = cv.erode(threshold,np.ones((3,3)))
    threshold = cv.dilate(threshold, np.ones((4, 4)))

    cv.imshow("After Preprocess", threshold)
    _, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    CNTS = []
    for c in contours:
       if cv.contourArea(c) < 700 and cv.contourArea(c) > 120:
           CNTS.append(c)

       print(len(CNTS))
    #cv.imshow("Bounding",test_image)
    counter = 0
    for cnt in CNTS:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0),2)
        perimeter = 2*(w+h)
        area = w*h
        compactness = (perimeter**2)/area

        M = cv.moments(cnt)


        u20 = (M['m20'] / M['m00']) - (M['m10'] / M['m00']) ** 2
        u02 = (M['m02'] / M['m00']) - (M['m01'] / M['m00']) ** 2
        u11 = (M['m11'] / M['m00']) - ((M['m10'] / M['m00']) * (M['m01'] / M['m00']))


        a = u20 + u02
        b = u11
        c = u20 - u02

        E_min = (a / 2) - ((c / 2) * (c / (math.sqrt(c ** 2 + b ** 2)))) - (
                    (b / 2) * (b / (math.sqrt(c ** 2 + b ** 2))))
        E_max = (a / 2) + ((c / 2) * (c / (math.sqrt(c ** 2 + b ** 2)))) + (
                    (b / 2) * (b / (math.sqrt(c ** 2 + b ** 2))))

        circularity = E_min / E_max

        # if(compactness>=17):
        #     cv.putText(test_image,"Open {:.2f}".format(compactness), (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 1)
        # else:
        #     cv.putText(test_image, "Closed {:.2f}".format(compactness), (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


        if (circularity >= 0.55):
            cv.putText(test_image, "Closed {:.2f}".format(circularity), (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 1)
        else:
            cv.putText(test_image, "Open {:.2f}".format(circularity), (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 1)

        counter = counter + 1



    cv.imshow("CIRCULARITY",test_image)



    #cv.imshow("Preprocessed", copy)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ =="__main__":
    main()