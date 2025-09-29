# import
import sys
from matplotlib.colors import cnames
import pandas as pd
import numpy as np
import os
from sqlalchemy import null
from tqdm import tqdm
import sklearn as sk
import tensorflow as tf
from pathlib import Path
import cv2
import pydicom
from PIL import Image



# function to find the mean value of the column
def fillWithMean(db,col_names = null):

    # print("pippo")
    if col_names == null:
        col_names = list(db.select_dtypes(exclude=['string','object','bool','datetime']).columns)

    print("numeric cols with nan values: "+str(col_names))
    for col in col_names:
        n = 0
        coln = pd.DataFrame(db[col])
        s = coln.sum().sum()
        n = len(coln)-coln.isnull().sum().sum()
        m = s/n
        db[col] = db[col].fillna(m)
    return db

# IMAGE LOADER

def loadImgTimeToArray(DB,d,cName,t,basePt):
    paths = DB[cName].to_list()
    # print(paths)
    imgs = []
    for p in tqdm(paths):
        p = p[1:-1]
        p = p.split(",")
        p = list(map(lambda x:x.split("'")[1], p))
        if len(p)>t:
            img = loader(basePt / Path(p[t]),d)
        imgs.append(img)

    imgs = np.array(imgs)
    return imgs

def loadAllImgTimeToArray(DB,d,cName,basePt):
    paths = DB[cName].to_list()
    # print(paths)
    imgs = []
    for p in tqdm(paths):
        p = p[1:-1]
        p = p.split(",")
        p = list(map(lambda x:x.split("'")[1], p))
        for pp in p:
            img = loader(basePt / Path(pp),d)
            imgs.append(img)

    imgs = np.array(imgs)
    return imgs

def clahe_transform(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply((img * 255).astype(np.uint8)) / 255
    return img

def normalize(img, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()
    img = (img - min_val) / (max_val - min_val)
    # img -= img.mean()
    # img /= img.std()
    return img

def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    if extension == ".dcm":
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array.astype(float)
        photometric_interpretation = dicom.PhotometricInterpretation
    else:
        img = Image.open(img_path)
        img = np.array(img).astype(float)
        photometric_interpretation = None
    return img, photometric_interpretation

# def loader(img_path, img_dim, mask_path=None, box=None, clahe=False):
def loader(img_path, img_dim, clahe=False):
    # Img
    img, photometric_interpretation = load_img(img_path)
    min_val, max_val = img.min(), img.max()

    # Pathometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)

    ## MASK AND BOX (to do?)
    # # Filter Mask
    # if mask_path:
    #     mask, _ = load_img(mask_path)
    #     img = get_mask(img, mask, value=1)

    # # Select Box Area
    # if box:
    #     img = get_box(img, box, perc_border=0.5)

    # Resize
    img = cv2.resize(img, (img_dim, img_dim))
    # Normalize
    img = normalize(img, min_val=min_val, max_val=max_val)
    # CLAHE
    if clahe:
        img = clahe_transform(img)
    
    # To 3 Channels
    img = np.stack((img, img, img), axis=-1)
    
    return img
