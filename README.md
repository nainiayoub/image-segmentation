# Contour-based segmentation
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nainiayoub/image-segmentation/main/app.py)

We want to perform the segmentation of a set of images by performing a contour-based segmentation on a reference proposed by an expert.
Description

Generally, image segmentation goes through 3 phases:
* Pre-processing (filtering)
* Processing (contour segmentation)
* Post-processing (morphology)

For each phase, there is a set of possible operators (methods or algorithms). Our task is to find and render the best operators that give a result very close to the ground truth. The tests are sometimes done on the operators and other times on their parameters.

Two types of images are given: `Simple Images (Signals)` and `Textured Images (Real Images)`.
