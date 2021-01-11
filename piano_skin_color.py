import cv2 as cv
import numpy as np
import os


##Load images and resize them
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv.resize(img, (700, 700))
            images.append(img)
    return images


#Convert to grayscale
def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   # _, img = cv.threshold(img, 40, 255, cv.THRESH_BINARY)
    return img


# Thresholding based on skin color.
def skin_color(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i, j]
            b, g, r = val[0], val[1], val[2]
            if ((r > 95 and g > 40 and b > 20) and ((max(r, g, b) - min(r, g, b, )) > 15) and (abs(r - g) > 15) and r > g and r > b):
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


# helper function to find contours
def find_contours(img):
    _, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def main():
    folder = "CS585-PianoImages"
    dataset = load_images_from_folder(folder)
    img = dataset[1]
    cv.imshow("Original Image", img)
    cv.imwrite("Piano_frame.jpg",img)
    mean_frame = np.mean(dataset, axis=0).astype(dtype=np.uint8)
    

    diff = cv.subtract(img, mean_frame)
    sk = skin_color(diff)
    preprocessed = preprocess(sk)

    split = preprocessed[:, :350]
    split = cv.dilate(split, np.ones((7, 7)))
    split = cv.erode(split,cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    cv.imshow("split", split)
    contours = find_contours(split)

    CNTS = []
    for c in contours:
        if cv.contourArea(c) > 40:
            CNTS.append(c)

    for cnt in CNTS:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x - w, y), (x + w, y + 2 * h), (0, 255, 0), 2)

    cv.imshow("Hands", img)
    cv.imwrite("Piano_hands.jpg",img)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()