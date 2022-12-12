import numpy as np
import cv2
import imagehash

from PIL import Image
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt

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

# Histogram intersection
def hintersection(pathA, pathB):
	imageA = cv2.imread(pathA)
	imageB = cv2.imread(pathB)

	histogramABlue = cv2.calcHist(imageA, [0], None, [200], [0, 256])
	histogramAGreen = cv2.calcHist(imageA, [1], None, [200], [0, 256])
	histogramARed = cv2.calcHist(imageA, [2], None, [200], [0, 256])

	histogramBBlue = cv2.calcHist(imageB, [0], None, [200], [0, 256])
	histogramBGreen = cv2.calcHist(imageB, [1], None, [200], [0, 256])
	histogramBRed = cv2.calcHist(imageB, [2], None, [200], [0, 256])

	cv2.normalize(histogramABlue, histogramBBlue, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(histogramAGreen, histogramAGreen, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(histogramARed, histogramARed, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	cv2.normalize(histogramBBlue, histogramBBlue, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(histogramBGreen, histogramBGreen, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(histogramBRed, histogramBRed, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	scoreBlue = cv2.compareHist(histogramABlue, histogramBBlue, method=cv2.HISTCMP_INTERSECT)
	scoreGreen = cv2.compareHist(histogramAGreen, histogramBGreen, method=cv2.HISTCMP_INTERSECT)
	scoreRed = cv2.compareHist(histogramARed, histogramARed, method=cv2.HISTCMP_INTERSECT)

	return scoreBlue + scoreGreen + scoreRed

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

# Comparison score in %
def compResults(pathA, pathB, method):
	return round((method(pathA, pathB) / method(pathA, pathA)) * 100)

if __name__ == '__main__':
	pathA = 'assets/image_a.jpg'
	pathB = 'assets/image_b.jpg'
	pathC = 'assets/image_c.jpg'
	pathD = 'assets/image_d.jpg'
	pathE = 'assets/image_e.jpg'
	pathF = 'assets/image_f.jpg'

	print(f'Image A is the reference image.')
	print(f'------')
	print(f'MSE: A<>B={mse(pathA, pathB)}, A<>C={mse(pathA, pathC)}, A<>D={mse(pathA, pathD)}, A<>E={mse(pathA, pathE)}, A<>F={mse(pathA, pathF)}')
	print(f'pHash: A<>B={phash(pathA, pathB)}, A<>C={phash(pathA, pathC)}, A<>D={phash(pathA, pathD)}, A<>E={phash(pathA, pathE)}, A<>F={phash(pathA, pathF)}')
	print(f'Intersection: A<>B={compResults(pathA, pathB, hintersection)}, A<>C={compResults(pathA, pathC, hintersection)}, A<>D={compResults(pathA, pathD, hintersection)}, A<>E={compResults(pathA, pathE, hintersection)}, A<>F={compResults(pathA, pathF, hintersection)}')
	print(f'SIFT: A<>B={sift(pathA, pathB)}, A<>C={sift(pathA, pathC)}, A<>D={sift(pathA, pathD)}, A<>E={sift(pathA, pathE)}, A<>F={sift(pathA, pathF)}')
	print(f'SSIM: A<>B={ssim(pathA, pathB)}, A<>C={ssim(pathA, pathC)}, A<>D={ssim(pathA, pathD)}, A<>E={ssim(pathA, pathE)}, A<>F={ssim(pathA, pathF)}')