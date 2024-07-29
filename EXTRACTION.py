#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pywt
import numpy as np
import scipy.fftpack
import os
import threading



# ## directory

# In[2]:


embedded_DIR = "embedded\\"
extracted_DIR ="extracted\\"


channel = 0 #0:blue/y,1:green/cg,2:red/co

def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        image_names.append(filename)
    return image_names


# ## for extracting middle portion from the cover or watermarked image's HH1's even and odd matrix

# In[3]:


def extract_middle_portion(A, m, n, x, y):
  #A is source matrix; m,n are dimension of A(passing this for optimization reasons);x,y dimention of watermark matrix
  if x > m or y > n:
    raise ValueError("x and y must be less than or equal to the corresponding dimensions of the matrix.")

  # Calculate the starting indices for the middle portion.
  start_row = (m - x) // 2
  start_col = (n - y) // 2

  # Extract the middle portion.
  return A[start_row:start_row+x, start_col:start_col+y]


# ## for reshaping watermark from 3d to 2d and vice versa

# In[4]:


def deconcatenate_2d_to_color_image(matrix, dims):

  # Reshape the matrix back into a 3D matrix.
  image = np.reshape(matrix, dims)

  # Return the deconcatenated image.
  return image



# ## converting cover and watermarked image from RGB Color space to YCbCr colorspace and vice versa (a bit better psnr than the cv2 implementaiton)

# In[5]:


def rgb_to_ycbcr_lossless(img_bgr):
  matrix = np.array([[0.299, 0.587, 0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]], dtype=np.float64)
  img_ycrcb = np.dot(img_bgr, matrix.T)
  if (channel == 0):
      return img_ycrcb
  return img_bgr



# ## for splitting matrix into even and odd pixel's location wise matrices and merging them(row wise or column wise is selected according to whichever is lager ,default is row wise) along with for edge cases where input image's dimensions are not in the power of 2

# In[6]:


def row_even_odd_split(array):
    
  even_array = array[::2, :]
  odd_array = array[1::2, :]
  return even_array, odd_array

def column_even_odd_split(array):

  even_array = array[:, ::2]
  odd_array = array[:, 1::2]
  return even_array, odd_array




#main function start from here 

def even_odd_split(matrix):
    flag = False
    last_elements = None
    if matrix.shape[0]>=matrix.shape[1]:#if row length > column length a or square matrix
        flag = True #flag is used to determine if the fucntion will run row wise or column wise
        if matrix.shape[0]%2 != 0: #for edge cases where shape is not in power of 2
            last_elements = matrix[-1, :]
            matrix = matrix[:-1, :]
        even, odd = row_even_odd_split(matrix)              
    else:
        #print(matrix.shape) #if row length < column length a
        if matrix.shape[1]%2 != 0:
            last_elements = matrix[:, -1]
            matrix = matrix[:, :-1]
        even, odd = column_even_odd_split(matrix)
        #print(matrix.shape)

    return even, odd, flag, last_elements


# ## embedding and extraction mathematical operations

# In[7]:


def recover_watermark(A,shape,alpha=0.1):#recover_watermark(A, rowW, colW, alpha=0.1):
    alpha = np.float64(alpha)
    two = np.float64(2.)
    
    Wrow = shape[0]
    Wcol = shape[1] * shape[2]
    
    evenw, oddw, flag, last= even_odd_split(A)
    
 

    roweven, coleven = evenw.shape
    
    
    ZZeven = extract_middle_portion(evenw.copy(), roweven, coleven, Wrow, Wcol)
    ZZodd = extract_middle_portion(oddw.copy(), roweven, coleven,Wrow, Wcol)

   
    

    # Initialize matrix W with zeros
    flattenW = np.zeros((Wrow,Wcol), dtype=np.float64)

    # Iterate through even indices of A and recover B using the specified equation

    flattenW = ((ZZeven - ZZodd)/two)
    flattenW= flattenW/alpha
    
    
    return flattenW


# # EXTRACTION

# In[8]:


def process4extract_image(image_name, embedded_DIR_local, extracted_DIR_local, shape):
  wimage_path = embedded_DIR_local+image_name
  image1 = cv2.imread(wimage_path)
  image = rgb_to_ycbcr_lossless(image1)
  bwm =image[:, :, channel]

  #for certain edge cases where dimension of image is not in power of 2
  if bwm.shape[0] % 4 != 0:
        rowLen=bwm.shape[0] % 4
        last_row = bwm[-rowLen:,:]
        bwm= bwm[:-rowLen,:]
  
  if bwm.shape[1] % 4 != 0:
        colLen=bwm.shape[1] % 4
        last_column = bwm[:,-colLen:]
        bwm= bwm[:,:-colLen]

  #bwm_LL, (bwm_LH, bwm_HL, bwm_HH) = pywt.dwt2(bwm, 'db1')
  values = pywt.wavedec2(bwm, 'coif3', mode='periodization', level=2)
  (b_LH, b_HL, b_HH) = values[1]


    
  water = recover_watermark(b_HH, shape)
  watermark_ext = deconcatenate_2d_to_color_image(water,shape)

  if not image_name.endswith(".png"):
    image_name = image_name.split(".")[0] + ".png"
  extracted_water_path = extracted_DIR_local+"extracted_from_"+image_name
  cv2.imwrite(extracted_water_path, watermark_ext)



# ## EXTRACT MULTITHREADED

# In[10]:


def extract(embedded_DIR_local=embedded_DIR, extracted_DIR_local=extracted_DIR, shape=(32,32,3)):
    """
    Extracts hidden images from embedded images using multithreading.

    This function processes all images in the specified embedded directory,
    extracts hidden images, and saves them in the extracted directory.
    It uses multithreading to improve performance.

    Parameters:
    embedded_DIR_local (str): Path to the directory containing embedded images.
                              Defaults to the global 'embedded_DIR'.
    extracted_DIR_local (str): Path to the directory where extracted images will be saved.
                               Defaults to the global 'extracted_DIR'.
    shape (tuple): The shape of the embedded image. Defaults to (32, 32, 3).

    Returns:
    None

    """
    
    embedd_image_names = get_image_names(embedded_DIR_local)
    # Create the folder if it doesn't exist
    os.makedirs(extracted_DIR_local, exist_ok=True)
    threads = []
    for imgname in embedd_image_names:
        thread = threading.Thread(target=process4extract_image, args=(imgname, embedded_DIR_local, extracted_DIR_local, shape))
        threads.append(thread)
    
    
    for thread in threads:
        thread.start()
    
    
    for thread in threads:
        thread.join()


# In[11]:


# %%time
# extract()

