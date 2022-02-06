import streamlit as st
import cv2
import os
import numpy as np
#----------------------
from skimage import io, img_as_float, color
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from scipy import ndimage

st.title("Contour based segmentation")
st.markdown("""
    
    We want to perform the segmentation of a set of images 
    by performing a contour-based segmentation on a reference proposed by an expert.
    
    ### Description
    Generally, image segmentation goes through 3 phases:
    * Pre-processing (filtering)
    * Processing (contour segmentation)
    * Post-processing (morphology)
    
    For each phase, there is a set of possible operators (methods or algorithms).
    Our task is to find and render the best operators that give a result very 
    close to the ground truth. The tests are sometimes done on the operators and other times on their parameters.

    
    Two types of images are given: `Simple Images (Signals)` and `Textured Images (Real Images)`.
""")

st.markdown("""
### Textured Images (Real Images)
The input images will have to be read in Grayscale
""")

real_images = './images/real-images'
signal_images = './images/signal-images'

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]
# displaying images and reading them grayscale

c = 0
images = []
images_gray = []
for img in os.listdir(real_images):
    if c >= 4:
        break
    img_path = real_images+"/"+img
    # origianls
    images.append(img_path)
    # Grayscale
    images_gray.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    # col = col+str(1)
    with cols[c]:
        st.image(img_path)

    c = c+1

# Display Grayscale images

for i in range(len(images_gray)):
    with cols[i]:
        st.image(images_gray[i])

# col1, col2, col3, col4 = st.columns(4)
# cols = [col1, col2, col3, col4]

# ---------------------------------- Pre-processing filtering -------------------------------------------------
def mean_filter(img):
    kernel = np.ones((3,3),np.float32)/9
    processed_image = cv2.filter2D(img,-1,kernel)
    return processed_image

# Median
def median_filter(img):
    processed_image = cv2.medianBlur(img, 3)
    return processed_image

# Gauss Filter
def gauss_filter(img):
    # read image
    img = img_as_float(img) # Load image
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    #Applay Gaussian
    gaussian_kernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]) #Gaussian Kernel
    conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel , borderType=cv2.BORDER_CONSTANT) 

    return conv_using_cv2

# bilateral
def bilateral_filter(img):
  blur = cv2.bilateralFilter(img,9,75,75)

  return blur

# Get filtered output images
filtered_images = []
for i in range(len(images)):
    img_filtered = []
    # mean
    mean_img = mean_filter(images_gray[i])
    img_filtered.append(mean_img)

    # median
    median_img = median_filter(images_gray[i])
    img_filtered.append(median_img)

    # Gaussian
    gauss_img = gauss_filter(images_gray[i])
    img_filtered.append(gauss_img)

    # Bilateral
    bil_img = bilateral_filter(images_gray[i])
    img_filtered.append(bil_img)

    # add to image
    filtered_images.append(img_filtered)

#-------------- Display Filtered Images --------------

with st.expander("Pre-processing: Filtering"):
    st.write("#### Pre-processing: Filtering")
    col8, col9, col10, col11 = st.columns(4)
    colsFil = [col8, col9, col10, col11]
    fils_desc = ['Mean Filter', 'Median Filter', 'Gaussian Filter', 'Bilateral Filter']
    for i in range(len(colsFil)):
        with colsFil[i]:
            st.write(fils_desc[i])
            for j in range(len(filtered_images)):
                st.image(filtered_images[j][i])
                

# ---------------------------------------------- Segmentation ----------------------------------------------

def tresholdImg(im, t):

    gray = np.copy(im)
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    mean = gray_r.mean()

    for i in range(gray_r.shape[0]):
        if gray_r[i] > t: # try with -0.1 in b.jpg image 
            gray_r[i] = 0
        else:
            gray_r[i] = 1
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])

    return gray

def laplacian_filter(img, t):
    rst = cv2.Laplacian(img, cv2.CV_64F)
    #Threshold the image
    result = tresholdImg(rst, t)
    return rst

def auto_canny(image, sigma=0.53):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def canny_filt(img):
    r = cv2.Canny(img, 10, 200)
    return r

# Get segmented output images
# tresh = st.slider("Treshhold value", 0, 255, 101)
# seg_images = []
# for i in range(len(filtered_images)):
#     img_seg = []
#     for j in range(len(filtered_images[i])):
#         # mean
#         laplacian = auto_canny(filtered_images[i][j])
#         img_seg.append(laplacian)
#         # add to image
#     seg_images.append(img_seg)

# st.write(filtered_images[0][0])



# with st.expander("Processing: Contour-based Segmentation"):
#     st.write("#### Processing: Contour-based Segmentation")
#     col12, col13, col14, col15 = st.columns(4)
#     colsSeg = [col12, col13, col14, col15]
#     for i in range(len(colsSeg)):
#         with colsSeg[i]:
#             st.write(fils_desc[i]+' and Laplacian')
#             for j in range(len(seg_images)):
#                 st.image(seg_images[j][i], clamp=True)

# tresh = st.slider("Treshhold value", 0, 255, 101)
# st.image(laplacian_filter(filtered_images[0][0], tresh))

#------------------------------------------------------------------------------------
def SegFun(path,t):
    # read image
    img = img_as_float(io.imread(path, as_gray=True)) # Load image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Applay Gaussian
    gaussian_kernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]) #Gaussian Kernel
    conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel , borderType=cv2.BORDER_CONSTANT) 
    
    # Apply Laplacian kernel
    kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    out_l = ndimage.convolve(conv_using_cv2, kernel_laplace, mode='reflect')

    #Normalize image
    normalized_img = ( (out_l - out_l.min())/( out_l.max()-out_l.min() ) )*255
    normalized_img = np.uint8(normalized_img)
    
    #Threshold the image
    result = tresholdImg(normalized_img, t)
#     result = np.uint8(result)
    return result

# ------------------------ edge detection
@st.cache
def edge_detection(image,low_thres,high_thresh):
    #convert the image to RGB 
    # image = cv2.imread(image, cv2.COLOR_BGR2RGB)

    # #convert the image to gray 
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edged = cv2.Canny(image, low_thres,high_thresh)

    return edged

# ----------- Compute Edges -----------
def dilation(img, iterat):
    kernel = np.ones((5,5),np.uint8)
    died = cv2.dilate(img, kernel, iterations=iterat)

    return died

with st.expander("Processing and Post-Processing"):
    st.write("#### Processing: Edge Detection")
    cola, colb, colg = st.columns(3)
    with cola:
        low_thres = st.slider('Lower threshold for edge detection',min_value = 0 , max_value = 240,value= 80)
    with colb:
        high_thresh = st.slider('High threshold for edge detection',min_value = 10, max_value =240,value = 100)
    with colg:
        iterations = st.slider('Dilation iterations', min_value=0, max_value=20, value=1)

    if low_thres > high_thresh:    
        high_thresh = low_thres +5

    colc, cold, cole, colf = st.columns(4)
    colEdges = [colc, cold, cole, colf]
    edges = []
    for i in range(len(colEdges)):
        with colEdges[i]:
            st.write(fils_desc[i])
            for j in range(len(filtered_images)):
                if i != 2:
                    edg = edge_detection(filtered_images[j][i], low_thres, high_thresh)
                    morpho_d = dilation(edg, iterations)
                    # edges.append
                    st.image(edg)
                    st.markdown("__Dilation__")
                    st.image(morpho_d)
                    st.markdown("-------------------------------------------------------------------------------")
    


# ------------------------------- Signalisation ---------------------------------------



st.markdown("""
    ### Signalisation images (Simple Images)
    Approach for simple images (signals).
    """)
colr, cols, colt, colu, colv, colx, coly, colz, colm, coln, colo, colp, colq = st.columns(13)
colx = [colr, cols, colt, colu, colv, colx, coly, colz, colm, coln, colo, colp, colq]
# displaying images and reading them grayscale



c = 0
images_sig_or = []
images_signals_gray = []
for img in os.listdir(signal_images):
    # if c >= 8:
    #     break
    img_path = signal_images+"/"+img
    # origianls
    images_sig_or.append(img_path)
    # Grayscale
    images_signals_gray.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    # col = col+str(1)
    with colx[c]:
        st.image(img_path)

    c = c+1

# Display Grayscale images

for i in range(len(images_signals_gray)):
    with colx[i]:
        st.image(images_signals_gray[i])

# Get filtered output images
filtered_images_sig = []
for i in range(len(images_sig_or)):
    img_filtered_sig = []
    # mean 
    mean_img = mean_filter(images_signals_gray[i])
    img_filtered_sig.append(mean_img)

    # median
    median_img = median_filter(images_signals_gray[i])
    img_filtered_sig.append(median_img)

    # Gaussian
    gauss_img = gauss_filter(images_signals_gray[i])
    img_filtered_sig.append(gauss_img)

    # Bilateral
    bil_img = bilateral_filter(images_signals_gray[i])
    img_filtered_sig.append(bil_img)

    # add to image
    filtered_images_sig.append(img_filtered_sig)

#-------------- Display Filtered Images --------------

with st.expander("Pre-processing: Filtering"):
    st.write("#### Pre-processing: Filtering")
    col8, col9, col10, col11 = st.columns(4)
    colsFil = [col8, col9, col10, col11]
    fils_desc = ['Mean Filter', 'Median Filter', 'Gaussian Filter', 'Bilateral Filter']
    for i in range(len(colsFil)):
        with colsFil[i]:
            st.write(fils_desc[i])
            for j in range(len(filtered_images_sig)):
                st.image(filtered_images_sig[j][i])


with st.expander("Processing and Post-Processing"):
    st.write("#### Processing: Edge Detection")
    cola, colb, colg = st.columns(3)
    with cola:
        low_thres = st.slider('Lower threshold for edge detection',min_value = 0 , max_value = 240,value= 80, key=0)
    with colb:
        high_thresh = st.slider('High threshold for edge detection',min_value = 10, max_value =240,value = 100, key=1)
    with colg:
        iterations = st.slider('Dilation iterations', min_value=0, max_value=20, value=1, key=2)

    if low_thres > high_thresh:    
        high_thresh = low_thres +5

    colc, cold, cole, colf = st.columns(4)
    colEdges = [colc, cold, cole, colf]
    edges = []
    for i in range(len(colEdges)):
        with colEdges[i]:
            st.write(fils_desc[i])
            for j in range(len(filtered_images_sig)):
                if i != 2:
                    edg = edge_detection(filtered_images_sig[j][i], low_thres, high_thresh)
                    morpho_d = dilation(edg, iterations)
                    # edges.append
                    st.image(edg)
                    st.markdown("__Dilation__")
                    st.image(morpho_d)
                    st.markdown("-------------------------------------------------------------------------------")