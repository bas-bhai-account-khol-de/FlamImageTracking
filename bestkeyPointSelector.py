print("this code selects and marks the best keypoints in an image and gives the coordinate in black and white image")
import tensorflow
import cv2
import numpy as np





matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
def getBestpoints(img_path,randomise = False):
    
        
    image= cv2.imread(img_path)
    if(randomise):
        image = randomiseImage(image)

    print("loaded Image " +str( image.shape) )
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp,desc =findkeypoints(img,False)
    randomiseImage(img,False)

    
    for i in range(10):
        randomImage  =  randomiseImage(img,False)
        kpt,desct = findkeypoints(randomImage,False)
        matches = matcher.knnMatch(desc, desct, k=2)
        nkp = []
        ndesc =[]
        good_matches = []
        ratio_thresh = 0.9  # Lowe's ratio test
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        for m in good_matches:
            nkp.append(kp[m.queryIdx])
            ndesc.append(desc[m.queryIdx])
        kp=tuple(nkp)
        desc = np.array(ndesc)
        print(len(kp))
        pass
    drawKp(image,img_path,kp,True)





def drawKp(img,img_path,kp ,show =False):
    kps,dsc = findkeypoints(img)
    keypoint_image = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # Display the image with keypoint
    if show:
        cv2.imshow('Warped', keypoint_image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if k != 27:
            print("path is "+img_path)
            getBestpoints(img_path,True)
            


def findkeypoints(gray_image , show =False):
    # orb = cv2.ORB_create(1000)
    orb = cv2.SIFT_create(500)
    # Detect keypoints
    keypoints ,  discriptor = orb.detectAndCompute (gray_image, None)
    # Draw keypoints
    keypoint_image = cv2.drawKeypoints(gray_image, keypoints, None, color=(0,255,0), flags=0)
    # Display the image with keypoints
    if show:
        cv2.imshow('Keypoints', keypoint_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return (keypoints ,  discriptor)


# create random homography of image
def randomiseImage(image ,show =False,shift =200):
    h, w = image.shape[:2]
    pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=float)
    pts_dst = pts_src + np.random.rand(4, 2) * shift - (shift/2)  # Adjust the range according to your needs
    H, _ = cv2.findHomography(pts_src, pts_dst)
    warped_image = cv2.warpPerspective(image, H, (w, h))
    if show:
        cv2.imshow('Warped', warped_image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if k != 27:
            randomiseImage(image,True)
    return warped_image






def test():
    getBestpoints("test/img/cinema.jpeg")



test()