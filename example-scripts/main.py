import numpy as np
import cv2
import imagehash

from PIL import Image
from skimage.metrics import structural_similarity

# Mean squared error
def mse(pathA, pathB):
	imageA = cv2.imread(pathA)
	imageB = cv2.imread(pathB)

	err = np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err

# Perceptual Hashing
def phash(pathA, pathB):
	hashA = imagehash.phash(Image.open(pathA))
	hashB = imagehash.phash(Image.open(pathB))

	return hashA - hashB

# Scale Invariant Feature Transform
def sift(pathA, pathB):
	imageA = cv2.imread(pathA)
	imageB = cv2.imread(pathB)

	sift = cv2.SIFT_create()

	# Keypoints + Descriptors
	kp1, des1 = sift.detectAndCompute(imageA, None)
	kp2, des2 = sift.detectAndCompute(imageB, None)

	# Matcher
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	goodMatches = []
	for m,n in matches:
		if m.distance < 0.75 * n.distance:
			goodMatches.append([m])

	# Calculate and return percent
	return (len(goodMatches)*100) / len(kp2)

# Structural Similarity Index
def ssim(pathA, pathB):
	imageA = cv2.imread(pathA)
	imageB = cv2.imread(pathB)

	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	return structural_similarity(grayA, grayB) * 100

if __name__ == '__main__':
	pathA = 'assets/bild_a.jpg'
	pathB = 'assets/bild_b.jpg'
	pathC = 'assets/bild_c.jpg'
	pathD = 'assets/bild_d.jpg'
	pathE = 'assets/bild_e.jpg'
	pathF = 'assets/bild_f.jpg'

	print(f'Image A is the reference image.')
	print(f'------')
	print(f'MSE: A<>B={mse(pathA, pathB)}, A<>C={mse(pathA, pathC)}, A<>D={mse(pathA, pathD)}, A<>E={mse(pathA, pathE)}, A<>F={mse(pathA, pathF)}')
	print(f'pHash: A<>B={phash(pathA, pathB)}, A<>C={phash(pathA, pathC)}, A<>D={phash(pathA, pathD)}, A<>E={phash(pathA, pathE)}, A<>F={phash(pathA, pathF)}')
	print(f'SIFT: A<>B={sift(pathA, pathB)}, A<>C={sift(pathA, pathC)}, A<>D={sift(pathA, pathD)}, A<>E={sift(pathA, pathE)}, A<>F={sift(pathA, pathF)}')
	print(f'SSIM: A<>B={ssim(pathA, pathB)}, A<>C={ssim(pathA, pathC)}, A<>D={ssim(pathA, pathD)}, A<>E={ssim(pathA, pathE)}, A<>F={ssim(pathA, pathF)}')