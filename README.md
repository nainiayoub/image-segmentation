# Contour-based segmentation
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nainiayoub/image-segmentation/main/app.py)



### Initial Demo
https://user-images.githubusercontent.com/50157142/152702603-94d2afea-aa48-4d42-8474-0a549b6adae2.mp4

## Description
We want to perform the segmentation of a set of images by performing a contour-based segmentation on a reference proposed by an expert.
Description

Generally, image segmentation goes through 3 phases:
* Pre-processing (filtering)
* Processing (contour segmentation)
* Post-processing (morphology)

For each phase, there is a set of possible operators (methods or algorithms). Our task is to find and render the best operators that give a result very close to the ground truth. The tests are sometimes done on the operators and other times on their parameters.

Two types of images are given: `Simple Images (Signals)` and `Textured Images (Real Images)`.
