import numpy as np
import pandas as pd
import matplotlib as plt
import cv2
import os
import pathlib
import inspect
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

# Variable that contain dataset directory for each class
# it's have to get the path more accurate that library can read the file each folder
# Each variables contains one rice classes so that we can just make a function and make the variable as a root directory to get
# the data set
arborio_dir = "C:/Users/naufa/Documents/mlInit/rice-GLCM/data/raw/Arborio"
basmati_dir = "C:/Users/naufa/Documents/mlInit/rice-GLCM/data/raw/Basmati"
ipsala_dir = "C:/Users/naufa/Documents/mlInit/rice-GLCM/data/raw/Ipsala"
jasmine_dir = "C:/Users/naufa/Documents/mlInit/rice-GLCM/data/raw/Jasmine"
karacadag_dir = "C:/Users/naufa/Documents/mlInit/rice-GLCM/data/raw/Karacadag"


# Function to generate glcm and extract the feature
# It also make the data frame for each classes
def extractImg(dir):
    # Example to make dataframe
    dir_name = os.path.basename(dir)
    df = pd.DataFrame(columns=["_contrast", "_homogeneity", "_energy", "_correlation"])

    # an empty list to keep the dictionary list
    extract_list = []

    trig = 0
    for filename in os.listdir(dir):
        # using opencv to track/read the image from directory path through the loop
        img = cv2.imread(os.path.join(dir, filename))
        # just to make sure the path is right and check if the image is empty or not
        if img is None:
            print("Error empty")
            break

        # to convert the image to grayscale image using opencv
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(
            grayImg,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            symmetric=True,
            normed=True,
        )  # declare and make a glcm matrix
        # get the feature from the matrix with skimage by props paramter
        contrast = graycoprops(glcm, prop="contrast")
        homogeneity = graycoprops(glcm, prop="homogeneity")
        energy = graycoprops(glcm, prop="energy")
        correlation = graycoprops(glcm, prop="correlation")

        # dictionary to keep as temporary place before append to dataframe
        ext_dict = {
            f"{dir_name}_contrast": contrast[0][0],
            f"{dir_name}_homogeneity": homogeneity[0][0],
            f"{dir_name}_energy": energy[0][0],
            f"{dir_name}_correlation": correlation[0][0],
        }
        extract_list.append(ext_dict)
        trig += 1
        if trig == 5:
            break

    return pd.DataFrame.from_dict(extract_list)


arborio_df = extractImg(arborio_dir)
print(arborio_df)
basmati_df = extractImg(basmati_dir)
print(basmati_df)
ipsala_df = extractImg(ipsala_dir)
print(ipsala_df)
jasmine_df = extractImg(jasmine_dir)
print(jasmine_df)
karacadag_df = extractImg(karacadag_dir)
print(karacadag_df)

frames = [arborio_df, basmati_df, ipsala_df, jasmine_df, karacadag_df]
# Variable that contain all the classes dataframe then merged into one dataframe
merged_df = pd.concat(frames, axis=1)
print(merged_df)
# merged_df.to_csv(r"C:/Users/naufa/Documents/mlInit/rice-GLCM/data/processed/rice-extract.csv")
