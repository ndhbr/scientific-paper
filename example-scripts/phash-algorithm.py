import numpy as np
import cv2

# Generates Perceptual Hash
def generatePerceptualHash(path, export=False):
    # Step 1: Read Image
    image = cv2.imread(path)
    if (export):
        cv2.imwrite('phash-results/phash-step-1.png', image)

    # Step 2: To grayscale	
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (export):
        cv2.imwrite('phash-results/phash-step-2.png', image)

    # Step 3: To 32x32
    image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_NEAREST)
    if (export):
        cv2.imwrite('phash-results/phash-step-3.png', image)

    # Step 4a: Discrete Cosinus Transform: Rows
    image = np.float32(image) / 255.0
    rows = cv2.dct(image, cv2.DCT_ROWS)
    
    # Step 4b: Discrete Cosinus Transform: Columns
    rows = cv2.rotate(rows, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rows = cv2.flip(rows, 0)
    cols = cv2.dct(rows, cv2.DCT_ROWS)
    image = cv2.flip(cols, 0)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if (export):
        cv2.imwrite('phash-results/phash-step-4.png', image * 255)

    # Step 5: Crop to 8x8
    image = image[0:8, 0:8]

    if (export):
        cv2.imwrite('phash-results/phash-step-5.png', image * 255)

    # Step 6a: Calculate median
    imageValues = image.flatten()
    median = np.median(imageValues)

    # Step 6b: intensity < median ? black : white
    for i in range(8):
        for j in range(8):
            if (image[i][j] < median):
                image[i][j] = 0.0
            else:
                image[i][j] = 1.0

    if (export):
        cv2.imshow('resultHash', image)
        cv2.imwrite('phash-results/phash-step-6.png', image * 255)

    # Build string binary perceptual hash
    hash = image.flatten()
    hash = map(int, hash)

    if (export):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ''.join(map(str, hash))

if __name__ == '__main__':
    pathA = 'assets/image_a.jpg'
    pathB = 'assets/image_b.jpg'
    
    # Generate hash for image A
    hashA = generatePerceptualHash(pathA, export=True)
    print(f'Hash A: {hashA}')

    # Generate hash for image B
    hashB = generatePerceptualHash(pathB)
    print(f'Hash B: {hashB}')

    # Hamming-Distance
    similarity = sum(hashA != hashB for hashA, hashB in zip(hashA, hashB))

    # Print result
    print(f'Hamming-Distance: A<>B: {similarity}')