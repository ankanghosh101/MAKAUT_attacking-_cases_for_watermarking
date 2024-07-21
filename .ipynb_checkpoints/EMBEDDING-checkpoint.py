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


images_DIR = "imagespng\\"
watermark_DIR = "watermark\\WaterM.png"
embedded_DIR = "embedded\\"



channel = 0 #0:blue/y,1:green/cg,2:red/co

def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        image_names.append(filename)
    return image_names


# ## for extracting/replacing middle portion from the cover or watermarked image's HH1's even and odd matrix

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


# In[4]:


def insert_middle_portion(A, m, n, x, y, middle_portion):
 
  #A is source matrix; m,n are dimension of A(passing this for optimization reasons);x,y dimention of watermark matrix
  if middle_portion.shape != (x, y):
    raise ValueError("The dimensions of the middle portion must be (x, y).")

  # Calculate the starting indices for the middle portion.
  start_row = (m - x) // 2
  start_col = (n - y) // 2

  # Create a copy of the original matrix to avoid modifying it in-place.
  new_A = A.copy()

  # Insert the middle portion into the new matrix.
  new_A[start_row:start_row+x, start_col:start_col+y] = middle_portion

  return new_A


# ## for reshaping watermark from 3d to 2d and vice versa

# In[5]:


def concatenate_color_image_to_2d(image):

  # Get the dimensions of the image.
  dims = image.shape

  # Reshape the image into a 2D matrix, with the third dimension as the first dimension.
  image = np.reshape(image, (dims[0], dims[1] * dims[2]))


  # Return the concatenated image.
  return image



# ## converting cover and watermarked image from RGB Color space to YCbCr colorspace and vice versa (a bit better psnr than the cv2 implementaiton)

# In[6]:


def rgb_to_ycbcr_lossless(img_bgr):
  matrix = np.array([[0.299, 0.587, 0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]], dtype=np.float64)
  img_ycrcb = np.dot(img_bgr, matrix.T)
  if (channel == 0):
      return img_ycrcb
  return img_bgr

def ycbcr_to_rgb_lossless(img_ycrcb):
  matrix = np.array([[1.0, 0.0, 1.402],[1.0, -0.344136, -0.714136],[1.0, 1.772, 0.0]], dtype=np.float64)
  img_bgr = np.dot(img_ycrcb, matrix.T)
  if (channel == 0):
      return img_bgr

  return img_ycrcb


# ## for splitting matrix into even and odd pixel's location wise matrices and merging them(row wise or column wise is selected according to whichever is lager ,default is row wise) along with for edge cases where input image's dimensions are not in the power of 2

# In[7]:


def row_even_odd_split(array):
    
  even_array = array[::2, :]
  odd_array = array[1::2, :]
  return even_array, odd_array

def column_even_odd_split(array):

  even_array = array[:, ::2]
  odd_array = array[:, 1::2]
  return even_array, odd_array

def row_even_odd_merge(even_array, odd_array):

  original_array = np.zeros((even_array.shape[0] * 2, even_array.shape[1]), dtype=even_array.dtype)
  original_array[::2, :] = even_array
  original_array[1::2, :] = odd_array
  return original_array

def column_even_odd_merge(even_array, odd_array):

  original_array = np.zeros((even_array.shape[0], even_array.shape[1]* 2), dtype=even_array.dtype)
  original_array[:, ::2] = even_array
  original_array[:, 1::2] = odd_array
  return original_array


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

def even_Odd_merge(even, odd, flag, last_elements):
    
    if flag:
        matrix = row_even_odd_merge(even, odd)
        if last_elements is not None:
            matrix = np.append(matrix, [last_elements], axis=0)
            
    else:
        matrix = column_even_odd_merge(even, odd)
        if last_elements is not None:
            matrix = np.append(matrix, last_elements[:, np.newaxis], axis=1)
            
    return matrix


# ## embedding mathematical operations

# In[8]:


def embedd_matrix(COVER, WATERMARK,alpha=0.1):
    alpha = np.float64(alpha)
    two = np.float64(2.)
    
    WATERMARK = np.float64(WATERMARK)
   
    WATERMARK *= alpha

    Wrow,Wcol = WATERMARK.shape
    
    even, odd, flag, last = even_odd_split(COVER)
    roweven, coleven = even.shape
   
    

    ZZeven = extract_middle_portion(even.copy(), roweven, coleven, Wrow, Wcol)
    ZZodd = extract_middle_portion(odd.copy(), roweven, coleven,Wrow, Wcol)


    
    Aeven = (ZZeven + ZZodd)/two
    Aodd = (ZZeven + ZZodd)/two

    Aeven += WATERMARK
    Aodd -= WATERMARK


    newEven = insert_middle_portion(even.copy(), roweven, coleven, Wrow, Wcol, Aeven)
    newOdd = insert_middle_portion(odd.copy(), roweven, coleven,Wrow, Wcol, Aodd)

    A = even_Odd_merge(newEven, newOdd, flag, last)
    
    
    return A



# # EMBEDDING

# In[9]:


def process4embedd_image(water, image_name, images_DIR_local, embedded_DIR_local):
     image_path = images_DIR_local+image_name
     image = cv2.imread(image_path)
     image = rgb_to_ycbcr_lossless(image)

     
     b =image[:,:,channel]

     #for certain edge cases where dimension of image is not in power of 2
     last_row = None 
     last_column = None


     if b.shape[0] % 4 != 0:
         rowLen=b.shape[0] % 4
         last_row = b[-rowLen:,:]
         b= b[:-rowLen,:]

     if b.shape[1] % 4 != 0:
         colLen=b.shape[1] % 4
         last_column = b[:,-colLen:]
         b= b[:,:-colLen]


    
     #wavedec2 documentation
        # https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html#d-multilevel-decomposition-using-wavedec2
     #waverec2 documentation
        # https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html#d-multilevel-reconstruction-using-waverec2
     #b_LL, (b_LH, b_HL, b_HH) = pywt.dwt2(b, 'db1')
     values = pywt.wavedec2(b, 'coif3', mode='periodization', level=2)
     (b_LH, b_HL, b_HH) = values[1]


     Nb_HH = embedd_matrix(b_HH, water)
        
     values[1] = (b_LH, b_HL, Nb_HH)

     b_inv_dwt = pywt.waverec2(values, 'coif3', mode='periodization')
     b_inv_dwt = np.clip(b_inv_dwt, 0, 255) 
     b_inv_dwt = np.uint8(b_inv_dwt)


     if last_column is not None: #combine back extracted row and column only for odd dim
         b_inv_dwt = np.concatenate((b_inv_dwt, last_column), axis=1)
     if last_row is not None: 
         b_inv_dwt = np.concatenate((b_inv_dwt, last_row), axis=0)

     # Combine channel-------------------------------------------------------
     image[:, :, channel] = b_inv_dwt
     image = ycbcr_to_rgb_lossless(image)
     embedded_image_path = embedded_DIR_local+"watermarked_image_"+image_name
     cv2.imwrite(embedded_image_path, image)


# ## EMBED MULTITHREADED

# In[11]:


def embed(images_DIR_local=images_DIR, embedded_DIR_local=embedded_DIR, watermark_DIR_local=watermark_DIR):
    image_names = get_image_names(images_DIR_local)
    
    watermark = cv2.imread(watermark_DIR_local)
    water = concatenate_color_image_to_2d(watermark)
    
    threads = []
    for imgname in image_names:
        thread = threading.Thread(target=process4embedd_image, args=(water, imgname, images_DIR_local, embedded_DIR_local))
        threads.append(thread)
    
    
    for thread in threads:
        thread.start()
    
    
    for thread in threads:
        thread.join()


# In[13]:


# %%time
# embed()

