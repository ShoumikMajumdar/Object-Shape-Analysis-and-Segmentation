import cv2 as cv
import numpy as np
import os

#Load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv.resize(img, (500, 500))
            images.append(img)
    return images

#Convert to grayscale and absolute thresholding
def preprocess(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img,80,255,cv.THRESH_BINARY)
    #img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return img


def main():
    folder = "CS585-BatImages/Gray"
    data = load_images_from_folder(folder)

    cv.imshow("1st image",data[22])
    test_image = data[22]
    cv.imwrite("bat_frame.jpg",test_image)
    threshold = preprocess(test_image)
    #threshold = cv.erode(threshold,np.ones((2,2)))
    copy = threshold.copy()
    cv.imshow("After Preprocess", threshold)
    _, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    CNTS = []
    for c in contours:
       if cv.contourArea(c) < 700 and cv.contourArea(c) > 10:
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

        if(compactness>=17):
            #cv.putText(test_image, "{:.2f}".format(compactness),(x,y+h+15),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv.putText(test_image,"Open {:.2f}".format(compactness), (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 1)
        else:
            cv.putText(test_image, "Closed {:.2f}".format(compactness), (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        counter = counter+1



    cv.imshow("boxes",test_image)
    cv.imwrite("bat_bounding.jpg",test_image)

    #cv.imshow("Preprocessed", copy)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ =="__main__":
    main()