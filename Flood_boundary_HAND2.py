import cv2 as cv
import numpy as np
import pandas as pd


def read_data():
    image = cv.imread("hand2.png")
    image = cv.resize(image, (300, 300))
    return image


def closing(img):
    # mask = np.ones((7,7))
    mask = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    img = cv.dilate(img, mask)
    img = cv.erode(img, mask)
    return img


def opening(img):
    mask = np.ones((15, 15))
    mask2 = np.ones((5, 5))
    img = cv.erode(img, mask)
    img = cv.dilate(img, mask)
    img = cv.dilate(img, mask2)
    return img


def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
    #img = cv.adaptiveThreshold(img,225,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,2)
    return img


def fill(data, start_coords, fill_value):
    xsize, ysize = data.shape
    orig_value = data[start_coords[0], start_coords[1]]

    stack = set(((start_coords[0], start_coords[1]),))

    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")

    while stack:
        x, y = stack.pop()
        if data[x, y] == orig_value:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))


# Maveshi istyle
def getCoord(data):
    cords = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i][j] == 255):
                return i, j


def getCont(cont):
    c_list = []
    for i in range(len(cont)):
        temp = np.array(cont[i][0, 0])
        data = temp[1], temp[0]
        c_list.append(data)
    return c_list


def runBoundary(counter, data, c_list):
    for item in c_list:
        print(len(c_list))
        x = item[0]
        y = item[1]
        boundary_fill(data, x, y)
        #cv.imshow("boundary-filled", data)
        # cv.imwrite("tissue_boundary.jpg",data)


def runFlood(data, c_list):
    value = 40
    for item in c_list:
        fill(data, item, value)
        value = value + 40
    cv.imshow("filled", data)
    # cv.imwrite("tissue_filled.jpg",data)
    cv.waitKey(0)
    cv.destroyAllWindows()


def boundary_fill(data, x, y):
    start_x = x
    start_y = y

    bx = x
    by = y
    bx_prev = bx
    by_prev = by
    firstIter = True
    counter = 0
    while (True):
        print("Start X,Y :   " + str(start_x) + " " + str(start_y))
        print("Current X, y:   " + str(x) + " " + str(y))
        counter += 1
        if counter == 4400:
            break
        print("Counter : " + str(counter))
        bx_prev = bx
        by_prev = by
        if firstIter:
            firstIter = False
        # else:
        #     if (x == start_x+1 and y == start_y):
        #         break

        if (bx == x and by == y):
            print("Start")
            bx = x - 1
            by = y
            if (check_obj(data, bx, by)):
                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x - 1 and by == y):
            print("West")
            bx = x - 1
            by = y - 1
            if (check_obj(data, bx, by)):

                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x - 1 and by == y - 1):
            print("South - West")
            bx = x
            by = y - 1
            if (check_obj(data, bx, by)):
                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x and by == y - 1):
            print("South")
            bx = x + 1
            by = y - 1
            if (check_obj(data, bx, by)):
                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x + 1 and by == y - 1):
            print("South - East")
            bx = x + 1
            by = y
            if (check_obj(data, bx, by)):

                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x + 1 and by == y):
            print(" East ")
            bx = x + 1
            by = y + 1
            if (check_obj(data, bx, by)):

                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x + 1 and by == y + 1):
            print("North - East")
            bx = x
            by = y + 1
            if (check_obj(data, bx, by)):

                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x and by == y + 1):
            print("North")
            bx = x - 1
            by = y + 1
            if (check_obj(data, bx, by)):

                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue

        if (bx == x - 1 and by == y + 1):
            print("North - West")
            bx = x - 1
            by = y
            if (check_obj(data, bx, by)):

                x = bx
                y = by
                bx = bx_prev
                by = by_prev
            continue


#        if(data[x][y] == 255):
#            break


def check_obj(data, a, b):
    if a >= 300 or b >=300 or a <= 0 or b <= 0:
        return True

    if data[a][b] == 80 or data[a][b] == 40 or data[a][b] == 120 or data[a][b] == 160 or data[a][b] == 200 or data[a][b] == 240:
        # if (not data[a][b] == 0) or (not data[a][b] == 255):
        data[a][b] = 255
        return True

    #return False


def get_skeleton(img):
    counter = 0

    img = img.copy()
    skel = img.copy()
    skel[:, :] = 0
    kernel = np.ones((5, 5))
    while True:
        # counter = counter+1
        # print(counter)
        eroded = cv.erode(img, kernel)
        temp = cv.dilate(eroded, kernel)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv.countNonZero(img) == 0:
            break

    return skel


def main():
    data = read_data()
    data = preprocess(data)
    #data = opening(data)
    #data = closing(data)
    # data = cv.medianBlur(data, 9)
    # data = cv.GaussianBlur(data,(3,3),0)

    data = cv.erode(data,np.ones((3,3)))
    data = cv.bitwise_not(data)
    cv.imshow("preprocessed", data)
    skeleton_image = data.copy()
    skeleton_image = get_skeleton(skeleton_image)
    cv.imshow("Skeleton", skeleton_image)
#    cv.imwrite("hand2_skeleton.jpg",skeleton_image)

    #contours = cv.Canny(data,100,200)
    #cv.imshow("Canny dikha", contours)
    _, contours, hierarchy = cv.findContours(data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    CNTS = []
    for c in contours:
        if cv.contourArea(c) > 20000:
            CNTS.append(c)

    print(len(CNTS))
    #cont_drawn = cv.drawContours(data,CNTS,-1,128,2)
    # contours = np.array(contours)
    c_list = getCont(CNTS)
    print(c_list)
    #print(data[254][0])
    runFlood(data, c_list)
    h,w = data.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv.floodFill(data,mask,c_list[0],128)
    cv.imshow("Flood",data)
    counter = 0
    runBoundary(counter, data, c_list)
    cv.imshow("Boundary",data)
    #cv.imwrite("hand2_Boundary.jpg",data)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()







