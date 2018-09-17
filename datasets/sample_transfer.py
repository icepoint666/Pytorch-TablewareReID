import os
import cv2

if __name__ == "__main__":
    rootdir = "/home/ubuntu/Program/Tableware/data/2018043000/样本/样本"
    train_save_dir = "../datas/dishes_dataset/train/"
    test_save_dir = "../datas/dishes_dataset/test/"
    class_list = os.listdir(rootdir)
    # prepare training data
    index = 10000
    label = 0
    for i in range(0, 40):
        label = label + 1
        path = os.path.join(rootdir, class_list[i])
        imgs_list = os.listdir(path)
        for i in range(0, len(imgs_list)):
            file_path = os.path.join(path, imgs_list[i])
            index = index + 1
            image = cv2.imread(file_path)
            cv2.imwrite(os.path.join(train_save_dir, str(index)+"_"+str(label)+".png"), image)

    # prepare test data
    index = 10000
    label = 0
    for i in range(0, len(class_list)):
        label = label + 1
        path = os.path.join(rootdir, class_list[i])
        imgs_list = os.listdir(path)
        for i in range(0, int(len(imgs_list)/4)):
            file_path = os.path.join(path, imgs_list[i])
            index = index + 1
            image = cv2.imread(file_path)
            cv2.imwrite(os.path.join(test_save_dir, str(index) + "_" + str(label) + ".png"), image)
