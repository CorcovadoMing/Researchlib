import cv2
import matplotlib.pyplot as plt


class _Match:
    def __init__(self):
        pass

    def distance(self, img1, img2, plot = False):
        orb = cv2.ORB_create(edgeThreshold = 10)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

        print('Keypoints found:', len(kp1), len(kp2))

        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x: x.distance)
        matches = matches[:5]
        dsum = 0
        for i in matches:
            dsum += i.distance

        if plot:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags = 2)
            plt.imshow(img3), plt.show()

        return dsum
