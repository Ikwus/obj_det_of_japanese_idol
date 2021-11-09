################
#
# Retinafaceでの推論高速化のために縦が800pxになるようにデータセットの大きさをそろえる
#
################

import cv2
import numpy
import glob
import os
import zipfile

def resize_height(_img, _size):
    _h, _w = _img.shape[:2]
    if _h > _w:
        _dst = cv2.resize(_img ,dsize = (round(_size * _w / _h), _size))
    else:
        _dst = cv2.resize(_img ,dsize = (_size, round(_size * _h / _w)))
    return _dst

def unzip(_path):
    _path_tmp = os.path.splitext(_path)
    if _path_tmp[1] == ".zip":
        print("Its zip")
        with zipfile.ZipFile(_path) as existing_zip:
            existing_zip.extractall(_path_tmp[0])
    return _path_tmp[0]

def read_image(_img):
    _img_src = cv2.imread(_img)
    if _img_src is None:
        import sys
        print('File import error, {} is not correct'.format(_img))
        sys.exit()
    return _img_src

def folder_check(_path):
    if _path[-1] != "/":
        _path = _path + "/"
    os.makedirs(_path, exist_ok=True)
    return _path

def resize_dataset(_path, _dst_folder, _size):
    _dst_folder = folder_check(_dst_folder)
    _path = unzip(_path)
    _image_path = glob.glob(_path+ "*.jpg")
    for _image in _image_path:
        _img_src = read_image(_image)
        _img_dst = resize_height(_img_src, _size)
        _dst_path = _dst_folder + os.path.basename(_image)
        cv2.imwrite(_dst_path, _img_dst)
    return 

if __name__ == "__main__":
    path = "raw_data/nijimasu_aeri-1189196006698573824(20191030_000316)-1288293300655415296(20200729_110032)-media.zip"
    dst_folder = "Dataset/aeri"
    resize_dataset(path, dst_folder, 800)
