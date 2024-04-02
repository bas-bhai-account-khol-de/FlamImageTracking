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

    for i in range(30):
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
    # drawKp(image,img_path,kp,True)
    return kp,desc





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
            getBestpoints(img_path,False)
            


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


def drawCircleAtKp(image,points,show=False):
    radius = 5  # Example radius
    # Specify the color of the circle in BGR (Blue, Green, Red)
    color = (0, 255, 0)  # Green

    # Specify the thickness of the circle line
    # Use -1 for thickness to fill the circle
    thickness = 2  # Example thickness
    # Draw the circle on the image

    
    for i in points:
        cv2.circle(image, tuple([int(i[0]),int(i[1])]), radius, color, thickness)
    return image



def createHomographies(image,kp,count =-1,show=True):
    n=len(kp)
    if count>0:
        n=count
    points   = []
    for i in range(n):
        points.append([kp[i].pt[0],kp[i].pt[1]])
    shift=400
    points = np.array(points,dtype=float)
    h, w = image.shape[:2]
    pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=float)
    pts_dst = pts_src + np.random.rand(4, 2) * shift - (shift/2)  # Adjust the range according to your needs
    H, _ = cv2.findHomography(pts_src, pts_dst)
    warped_image = cv2.warpPerspective(image, H, (w, h))
    transformed_points = cv2.perspectiveTransform(np.array([points], dtype=np.float32), H)
    # print("number to draw is " + str(transformed_points.shape))
    drawnImage = drawCircleAtKp(warped_image,transformed_points[0],True)
    if show:
        print("showing")
        cv2.imshow('Warped', drawnImage)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if k != 27:
            createHomographies(image,kp,count,show)



def test():
    kp ,d = getBestpoints("test/img/cinema.jpeg")
    image  = cv2.imread("test/img/cinema.jpeg")
    createHomographies(image,kp,-1,True)




test()