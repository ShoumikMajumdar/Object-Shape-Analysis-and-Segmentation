import cv2 as cv
import numpy as np
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv.resize(img, (500, 500))
            images.append(img)
    return images

def preprocess(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img,60,255,cv.THRESH_BINARY)
    return img

def main():
    folder = "CS585-PeopleImages/"
    dataset = load_images_from_folder(folder)
    #first = dataset[80]
    first = np.mean(dataset,axis=0).astype(dtype=np.uint8)
    img = dataset[92]
    cv.imshow("data",img)
    cv.imwrite("Pedestrian_frame.jpg",img)
    #cv.imshow("first", first)
    diff = cv.subtract(first,img)
    threshold = preprocess(diff)


    threshold = cv.morphologyEx(threshold,cv.MORPH_OPEN,np.ones((3,3)))
    #threshold = cv.dilate(threshold,np.ones((7,7)))
    #cv.imshow("Closing",threshold)
    _, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    CNTS = []
    for c in contours:
        if cv.contourArea(c) > 55:
            CNTS.append(c)


    #cont_drawn = cv.drawContours(img, CNTS, -1, (0, 255, 255), 2)
    #cv.imshow("Contours",cont_drawn)
    print(len(CNTS))

    counter = 1
    cv.putText(img, "Number of People = " + str(len(CNTS)), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for cnt in CNTS:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        counter = counter+1


    cv.imshow("Bounding boxes",img)
    cv.imwrite("Pedestrian_bounding.jpg",img)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__=="__main__":
    main()