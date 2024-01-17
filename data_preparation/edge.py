import os
import cv2 as cv

if __name__ == "__main__":
    path = "../LEVIR-CD-256/label/"
    pathout = "../LEVIR-CD-256/label_edge/"
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    ims_list = os.listdir(path)
    ims_list.sort()
    for im_list in ims_list:
        image = cv.imread(path + im_list, flags=-1)
        edges = cv.Canny(image, 100, 200)
        cv.imwrite(pathout+im_list, edges)
