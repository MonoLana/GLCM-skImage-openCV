import numpy as np
import pandas as pd
import matplotlib as plt
import cv2
import os
import pathlib
import inspect
from dotenv import load_dotenv
from natsort import natsorted
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

# Variable that contain dataset directory for each class
# it's have to get the path more accurate that library can read the file each folder
# Each variables contains one rice classes so that we can just make a function and make the variable as a root directory to get
# the data set

load_dotenv()
arborio_dir = os.getenv("ARBORIO_DIR")
basmati_dir = os.getenv("BASMATI_DIR")
ipsala_dir = os.getenv("IPSALA_DIR")
jasmine_dir = os.getenv("JASMINE_DIR")
karacadag_dir = os.getenv("KARACADAG_DIR")


# Function to generate glcm and extract the feature
# It also make the data frame for each classes
def extractImg(dir):
    # Example to make dataframe
    dir_name = os.path.basename(dir)

    # an empty list to keep the dictionary list
    extract_list = []

    # Trigger to stop the function just to test file sso you dont have to wait an hours to see the result
    trig = 0

    # Using natsorted from natsort to sort the confusing file looping
    files = natsorted(os.listdir(dir))

    for filename in files:
        # using opencv to track/read the image from directory path through the loop
        img = cv2.imread(os.path.join(dir, filename))
        # just to make sure the path is right and check if the image is empty or not
        if img is None:
            print("Error empty")
            break

        print("extracting: ", filename)

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
            f"{dir_name}_0_contrast": contrast[0][0],
            f"{dir_name}_45_contrast": contrast[0][1],
            f"{dir_name}_90_contrast": contrast[0][2],
            f"{dir_name}_135_contrast": contrast[0][3],
            f"{dir_name}_0_homogeneity": homogeneity[0][0],
            f"{dir_name}_45_homogeneity": homogeneity[0][1],
            f"{dir_name}_90_homogeneity": homogeneity[0][2],
            f"{dir_name}_135_homogeneity": homogeneity[0][3],
            f"{dir_name}_0_energy": energy[0][0],
            f"{dir_name}_45_energy": energy[0][1],
            f"{dir_name}_90_energy": energy[0][2],
            f"{dir_name}_135_energy": energy[0][3],
            f"{dir_name}_0_correlation": correlation[0][0],
            f"{dir_name}_45_correlation": correlation[0][1],
            f"{dir_name}_90_correlation": correlation[0][2],
            f"{dir_name}_135_correlation": correlation[0][3],
        }
        extract_list.append(ext_dict)
        # Uncomment the trigger if you wanna extract all the image
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
# Uncomment the function below to convert the dataframe to csv
merged_df = pd.concat(frames, axis=1)
# print(merged_df)
# merged_df.to_csv(r"C:/Users/naufa/Documents/mlInit/rice-GLCM/data/processed/rice-extract.csv")
