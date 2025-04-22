
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def subplots(imgs, nc=2, figsize=(10,5), titles=None):
    nr = int(np.ceil(len(imgs)/nc))
    plt.figure(figsize=figsize)
    for i,img in enumerate(imgs):
        plt.subplot(nr, nc, i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show(img, title=None):
    plt.figure(figsize=(10,5))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def detect_and_match_features(img1_gray, img2_gray):
    # Feature detection and description
    descriptor = cv2.ORB_create(5000)
    kps1, features1 = descriptor.detectAndCompute(img1_gray, None)
    kps2, features2 = descriptor.detectAndCompute(img2_gray, None)

    # Feature matching with ratio test for better matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(features1, features2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # If not enough good matches, use regular matching
    if len(good_matches) < 10:
        matches = bf.match(features1, features2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:100]

    return kps1, kps2, good_matches

def compute_homography(kps1, kps2, good_matches):
    # Get matched keypoints
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    # Compute homography with RANSAC
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H, status

def stitch_pair(img1, img1_gray, img2, img2_gray):
    # Detect and match features
    kps1, kps2, good_matches = detect_and_match_features(img1_gray, img2_gray)

    # Draw matches for visualization
    matched_img = cv2.drawMatches(img1, kps1, img2, kps2, good_matches[:50], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Compute homography
    H, status = compute_homography(kps1, kps2, good_matches)

    # Calculate dimensions for the warped image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of img1
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])

    # Transform corners through homography
    corners1_transformed = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), H)

    # Get combined x,y extents
    all_corners = np.concatenate((corners1_transformed,
                                 np.float32([[[0, 0]], [[0, h2]], [[w2, h2]], [[w2, 0]]])))

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp img1 to img2's plane with expanded canvas
    output_img = cv2.warpPerspective(img1, H_translation.dot(H),
                                    (x_max - x_min, y_max - y_min))

    # Place img2 on the canvas
    output_img[translation_dist[1]:h2+translation_dist[1],
               translation_dist[0]:w2+translation_dist[0]] = img2

    return output_img, matched_img

def stitch_multiple_images(images, grayscale_images):
    """
    Stitch multiple images in sequence

    Args:
        images: List of color images
        grayscale_images: List of grayscale versions of the images

    Returns:
        Stitched panorama
    """
    # Start with the first image
    result = images[0]
    result_gray = grayscale_images[0]

    # Store all intermediate results and match visualizations
    results = [result]
    match_visualizations = []

    # Sequentially stitch each image
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}...")
        result, matched_img = stitch_pair(images[i], grayscale_images[i], result, result_gray)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Save intermediate results
        results.append(result)
        match_visualizations.append(matched_img)

    return result, results, match_visualizations

def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Option 1: Load specific images
    # image_paths = ['/content/g1.png', '/content/g2.png', '/content/g3.png']

    # Option 2: Load all images from a directory
    image_paths = sorted(glob('/content/*.png'))  # Adjust pattern as needed

    # Load images
    images = []
    grayscale_images = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}")
            continue
        images.append(img)
        grayscale_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Display input images
    print(f"Loaded {len(images)} images")
    subplots(images, nc=min(3, len(images)), figsize=(15, 5),
             titles=[f"Image {i+1}" for i in range(len(images))])

    # Try OpenCV's built-in stitcher first
    try:
        stitcher = cv2.createStitcher() if int(cv2.__version__[0]) < 4 else cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)

        if status == 0:
            show(panorama, title="Panorama using OpenCV Stitcher")
            cv2.imwrite("output/panorama_opencv.jpg", panorama)
        else:
            print(f"OpenCV stitcher failed (status code: {status}), falling back to custom implementation")
            raise Exception("Stitcher failed")

    except Exception as e:

        # Custom stitching implementation
        final_panorama, intermediate_results, match_visualizations = stitch_multiple_images(images, grayscale_images)

        # Display feature matches
        if match_visualizations:
            subplots(match_visualizations, nc=1, figsize=(15, 10),
                     titles=[f"Matches: Image {i+2} → Previous Result" for i in range(len(match_visualizations))])

        # Display intermediate results
        if len(intermediate_results) > 1:
            subplots(intermediate_results, nc=1, figsize=(15, 10),
                     titles=[f"After stitching {i+1} images" for i in range(len(intermediate_results))])

        # Display final panorama
        show(final_panorama, title="Final Panorama (Custom Implementation)")

        # Save final panorama
        cv2.imwrite("output/panorama_custom.jpg", final_panorama)
        print("Panorama saved to output/panorama_custom.jpg")

if __name__ == "__main__":
    main()