import mediapipe as mp
import cv2
import numpy as np

def sobel_filter(gray):
    gray_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gray_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(np.square(gray_x) + np.square(gray_y)).astype(np.uint8)

if __name__ == "__main__":
    org_img = cv2.imread(r"media\myLeftHand.jpg")
    org_img_h,org_img_w,_ = org_img.shape
    # cv2.imshow("org_img",org_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True,
        max_num_hands=1, # 検出する手の数（最大2まで）
        min_detection_confidence=0.5) as hands:
        results = hands.process(org_img)
    landmarks = results.multi_hand_landmarks[0]
    palm_landmark_index = [0,1,2,5,9,13,17]
    palm_landmark_points = []
    for i in palm_landmark_index:
        landmark = landmarks.landmark[i]
        palm_landmark_points.append([landmark.x,landmark.y,landmark.z])
    palm_landmark_points = np.array(palm_landmark_points)
    palm_img_rect = [
        np.min(palm_landmark_points[:,0])*org_img_w,
        np.min(palm_landmark_points[:,1])*org_img_h,
        np.max(palm_landmark_points[:,0])*org_img_w,
        np.max(palm_landmark_points[:,1])*org_img_h,
    ]
    palm_img_rect = np.array(palm_img_rect).astype(int)
    xmin,ymin,xmax,ymax = palm_img_rect
    margin = 0
    xmin = np.clip(xmin-margin, 0, org_img_w)
    xmax = np.clip(xmax+margin, 0, org_img_w)
    ymin = np.clip(ymin-margin, 0, org_img_h)
    ymax = np.clip(ymax+margin, 0, org_img_h)
    print(palm_img_rect)
    palm_org_img = org_img[ymin:ymax,xmin:xmax,:]
    # palm_nlm_img = cv2.fastNlMeansDenoisingColored(palm_org_img,None,10,10,7,21)

    gray = cv2.cvtColor(palm_org_img,cv2.COLOR_BGR2GRAY)
    sobel_img = sobel_filter(gray)
    # nlm_sobel_img = sobel_filter(cv2.cvtColor(palm_nlm_img,cv2.COLOR_BGR2GRAY))
    cv2.imshow("sobel_img",sobel_img)
    # cv2.imshow("nlm_sobel_img",nlm_sobel_img)
    cv2.imshow("palm_org_img",palm_org_img)
    # cv2.imshow("palm_nlm_img",palm_nlm_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()