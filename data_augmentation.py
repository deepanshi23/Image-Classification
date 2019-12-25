##Importing the required libraries
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rescale
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import misc
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import glob
from skimage.transform import rotate
from skimage import util
from skimage.transform import warp,AffineTransform
from skimage import exposure
from skimage import io
from skimage.color import rgb2gray

##Functions for image augmentation
def random_rotation(original_image):
    random_degree = np.random.uniform(-45, 45)
    image_with_rotation = rotate(original_image, random_degree)
    return image_with_rotation

def horizontal_flip(original_image):
    horizontal_flip_image = original_image[:, ::-1]
    return horizontal_flip_image

def vertical_flip(original_image):
    vertical_flip_image = original_image[::-1, :]
    return vertical_flip_image

def random_noise(original_image):
    randomnoise_image = util.random_noise(original_image)
    return randomnoise_image

def rescale(original_image):
    random_scalex = np.random.uniform(0.8, 1.5)
    random_scaley = np.random.uniform(0.8,1.5)
    tform = AffineTransform(scale=(random_scalex, random_scaley))
    rescaled_image = warp(original_image, inverse_map=tform.inverse)
    return rescaled_image

def shearing(original_image):
    shear_factor = np.random.uniform(-0.5,0.5)
    tform = AffineTransform(shear = shear_factor)
    sheared_image = warp(original_image, inverse_map=tform.inverse)
    return sheared_image

def translation(original_image):
    translate_x = np.random.uniform(-50,50)
    translate_y = np.random.uniform(-50,50)
    tform = AffineTransform(translation=(translate_x, translate_y))
    translated_image = warp(original_image, inverse_map=tform.inverse)
    return translated_image

def contrast_stretching(original_image):
    p2, p98 = np.percentile(original_image, (2, 98))
    img_rescale = exposure.rescale_intensity(original_image, in_range=(p2, p98))
    return img_rescale

def grayscale(original_image):
    gray_image = rgb2gray(original_image)
    return gray_image

available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'vertical_flip': vertical_flip,
    'rescale': rescale,
    'shearing': shearing,
    'translation': translation,
    'contrast_stretching':contrast_stretching,
    'grayscale': grayscale
}

num_generated_files = 0

##Code to apply transformations randomly on images
for i in range(1,7):
    folder_path = "C://Users//Documents//earthquake_image_project//data//augmented_image_run%s" %(i)
    for image_path in glob.glob("C://Users//Documents//earthquake_image_project//data//*.jpg"):
        image_to_transform = misc.imread(image_path)
        num_transformations_to_apply = random.randint(1,len(available_transformations))
        num_transformations = 0
        transformed_image = image_to_transform
        while num_transformations <= num_transformations_to_apply:
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](transformed_image)
            num_transformations+=1
        num_generated_files+=1
        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)
        io.imsave(new_file_path, transformed_image)

    for image_path in glob.glob("C://Users//Documents//earthquake_image_project//data//*.jpeg"):
        image_to_transform = misc.imread(image_path)
        num_transformations_to_apply = random.randint(1,len(available_transformations))
        num_transformations = 0
        transformed_image = image_to_transform
        while num_transformations <= num_transformations_to_apply:
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](transformed_image)
            num_transformations+=1
        num_generated_files+=1
        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)
        io.imsave(new_file_path, transformed_image)

