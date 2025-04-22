# image-stitching-panorama
Panorama image generation using Image Stitching with ORB 

# Image Stitching for Panoramic Image Generation
This Python project implements an image stitching algorithm to generate panoramic images from a series of overlapping photographs. The project uses OpenCV for feature detection, matching, and homography computation to align and combine multiple images into a single wide-angle panorama.

# Features
Feature Detection & Matching: The code uses the ORB (Oriented FAST and Rotated BRIEF) algorithm for detecting keypoints and computing feature descriptors. It then matches features between images using a brute-force matcher with a ratio test to filter out weak matches.
Homography Calculation: The homography matrix is computed using RANSAC to robustly estimate the transformation that aligns the images.
Sequential Stitching: The images are stitched one by one, each new image being aligned with the previous result to create a seamless panorama.
Custom and OpenCV Stitcher: The code first attempts to use OpenCV's built-in stitching algorithm. If that fails, it falls back on a custom stitching approach that allows more control over intermediate steps, such as feature matching and homography computation.
Visualization: Intermediate results, feature matches, and the final panorama are displayed using matplotlib. This helps in understanding the feature matching process and inspecting the final stitched result.

# How It Works
Image Loading: Images are loaded from a specified directory, and each image is converted into grayscale for feature detection.
Feature Matching: For each pair of images, the algorithm detects and matches features using ORB, followed by applying a ratio test to retain the best matches.
Homography Calculation: A homography matrix is computed using the matched feature points, which is then used to warp and align the images.
Stitching: The images are sequentially stitched together using the computed homographies and a translation matrix to ensure all images fit into a single canvas.
Error Handling: If OpenCV's built-in stitcher fails, the code gracefully falls back to the custom stitching implementation to produce the panorama.
Output: The final panorama is saved to disk and displayed. Additionally, intermediate images showing feature matches and results of intermediate stitching steps are also visualized.

# Requirements
Python 3.x
OpenCV (cv2)
NumPy
Matplotlib
