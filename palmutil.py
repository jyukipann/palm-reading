import mediapipe as mp
import cv2
import numpy as np


def sobel_filter(gray):
    k = 3
    gray_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gray_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    return np.sqrt(np.square(gray_x) + np.square(gray_y)).astype(np.uint8)

# mediapipeにより、手のキーポイントを検出して手のひらのみの画像を出力する
def crop_palm_img(img):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(img)
    landmarks = results.multi_hand_landmarks[0]
    palm_landmark_index = [0, 1, 2, 5, 9, 13, 17]
    palm_landmark_points = []
    # copy_img = img.copy()
    img_h, img_w, _ = img.shape
    for i in palm_landmark_index:
        landmark = landmarks.landmark[i]
        palm_landmark_points.append([landmark.x, landmark.y, landmark.z])
        # copy_img = cv2.circle(copy_img,(int(landmark.x*img_w),int(landmark.y*img_h)),10,(255,0,0),-1)

    palm_landmark_points = np.array(palm_landmark_points)
    # print(palm_landmark_points)

    # _palm_landmark_points = np.copy(palm_landmark_points)
    palm_landmark_points[:, 0] *= img_w
    palm_landmark_points[:, 1] *= img_h
    # copy_img = cv2.drawContours(copy_img, _palm_landmark_points[:,[0,1]].reshape((1,-1,2)).astype(np.float32), -1, color=(0, 0, 255), thickness=2)
    rect = cv2.minAreaRect(
        palm_landmark_points[:, [0, 1]].reshape((1, -1, 2)).astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # copy_img = cv2.drawContours(copy_img,[box],0,(0,0,255),2)
    # cv2.imshow("palm landmark",cv2.resize(copy_img,(400,400)))
    palm_img_size = (500, 500)
    original_points = box.astype(np.float32)
    original_points = original_points[[1, 2, 3, 0]]
    margin = int(sum(palm_img_size)/2)/10
    transform_points = np.array([
        [margin, margin],
        [palm_img_size[0]-margin, margin],
        [palm_img_size[0]-margin, palm_img_size[1]-margin],
        [margin, palm_img_size[1]-margin],
    ], np.float32)
    M = cv2.getPerspectiveTransform(original_points, transform_points)
    palm_img = cv2.warpPerspective(img, M, palm_img_size)
    return palm_img


if __name__ == "__main__":

    org_img = cv2.imread(r"media\myLeftHand.jpg")

    # crop palm
    palm_org_img = crop_palm_img(org_img)
    cv2.imshow("palm_org_img", palm_org_img)

    # gray
    palm_gray = cv2.cvtColor(palm_org_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("palm_gray", palm_gray)
    palm_gray = cv2.equalizeHist(palm_gray)
    cv2.imshow("palm_gray_eq", palm_gray)

    # sobel
    sobel_img = sobel_filter(palm_gray)
    # sobel_img = cv2.equalizeHist(sobel_img)
    cv2.imshow("sobel_img", sobel_img)

    # nlm
    palm_sobel_nlm_img = cv2.fastNlMeansDenoising(sobel_img, None, 20, 10, 7)
    cv2.imshow("palm_sobel_nlm_img", palm_sobel_nlm_img)

    # threshold
    th = 95
    th, palm_sobel_nlm_th_img = cv2.threshold(
        palm_sobel_nlm_img, th, 255, cv2.THRESH_BINARY)
    # th, palm_sobel_nlm_th_img = cv2.threshold(
    #     palm_sobel_nlm_img, 0, 255, cv2.THRESH_OTSU)
    print(th)
    cv2.imshow("palm_sobel_nlm_th_img", palm_sobel_nlm_th_img)

    # morphology
    k = 3
    # kernel = np.ones((k,k),np.uint8)
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], np.uint8)
    palm_sobel_nlm_th_morph_img = palm_sobel_nlm_th_img.copy()
    palm_sobel_nlm_th_morph_img = cv2.morphologyEx(
        palm_sobel_nlm_th_morph_img, cv2.MORPH_CLOSE, kernel)
    # palm_sobel_nlm_th_morph_img = cv2.morphologyEx(
    #     palm_sobel_nlm_th_morph_img, cv2.MORPH_OPEN, kernel)
    # palm_sobel_nlm_th_morph_img = cv2.morphologyEx(palm_sobel_nlm_th_morph_img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("palm_sobel_nlm_th_morph_img", palm_sobel_nlm_th_morph_img)

    # thinning
    palm_sobel_nlm_th_morph_thinning_img = cv2.ximgproc.thinning(palm_sobel_nlm_th_morph_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    cv2.imshow("palm_sobel_nlm_th_morph_thinning_img", palm_sobel_nlm_th_morph_thinning_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
