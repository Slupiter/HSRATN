from PIL import Image

import numpy as np
import os


def get_path(path):
    '''返回目录中所有文件名列表'''
    return [os.path.join(path, f) for f in os.listdir(path)]

def mask(img,imgName,imgPath,size = (128,32),interpolation=Image.BICUBIC,mask=True):

    #img = img.resize(size, interpolation)
    #img.save(imgPath+'/'+imgName+'(128,32)'+'.jpg')
    if mask:
        mask = img.convert('L')
        thres = np.array(mask).mean()
        mask = mask.point(lambda x: 0 if x > thres else 255)
        mask.save(imgPath+'/'+imgName+'mask(128,32)'+'.jpg')

def main(imgPath='./makePic/maskPic'):
    imgList = get_path(imgPath)
    print(imgList)
    for i in imgList:
        img = Image.open(i)  # 读取文件
        filePath, fileName = os.path.split(i)
        imgName,_ = fileName.split('.')

        mask(img, imgName,imgPath)

if __name__ == '__main__':
    main()

