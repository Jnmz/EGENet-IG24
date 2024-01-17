# 此代码用于切分数据集


import cv2
import os


# 根据任务要求，定义一个caijian函数
def caijian(path, path_out, size_w=512, size_h=512, step=256):  # step为步长，设置为256即相邻图片重叠50%
    name = "img"

    img = cv2.imread(path, flags=-1)  # 读取要切割的图片,
    size = img.shape
    i = 0
    for h in range(0, size[0], step):
        star_h = h  # star_h表示起始高度，从0以步长step=256开始循环
        for w in range(0, size[1], step):
            star_w = w  # star_w表示起始宽度，从0以步长step=256开始循环
            end_h = star_h + size_h  # end_h是终止高度

            if end_h > size[0]:  # 如果边缘位置不够512的列
                # 以倒数512形成裁剪区域
                star_h = size[0] - size_h
                end_h = star_h + size_h
                i = i - 1
            end_w = star_w + size_w  # end_w是中止宽度
            if end_w > size[1]:  # 如果边缘位置不够512的行
                # 以倒数512形成裁剪区域
                star_w = size[1] - size_w
                end_w = star_w + size_w
                i = i - 1

            cropped = img[star_h:end_h, star_w:end_w]  # 执行裁剪操作
            i = i + 1
            name_img = name + '_' + str(star_h) + '_' + str(star_w)  # 用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
            #                 name_img = name + str(i)

            cv2.imwrite('{}/{}.png'.format(path_out, name_img), cropped)  # 将切割得到的小图片存在path_out路径下
    print("已成功执行！")


# 将完整的图像划分为小块
if __name__ == '__main__':
    ims_path = []
    ims_path.append('../WHU-CD/1. The two-period image data/before/before.tif')
    ims_path.append('../WHU-CD/1. The two-period image data/after/after.tif')
    ims_path.append('../WHU-CD/1. The two-period image data/change label/change_label.tif')

    path = []
    path.append('../WHU256/A')
    path.append('../WHU256/B')
    path.append('../WHU256/label')
    for i in range(len(path)):

        if not os.path.exists(path[i]):
            os.makedirs(path[i])
        caijian(ims_path[i], path[i], size_w=256, size_h=256, step=256)
