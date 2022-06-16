import mediapipe as mp
import cv2
import numpy as np

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
    margin = 100
    xmin -= margin
    np.clip(xmin-margin, 2, None)
    print(palm_img_rect)
    palm_org_img = org_img[ymin:ymax,xmin:xmax,:]
    cv2.imshow("palm_org_img",palm_org_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()