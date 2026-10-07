import cv2
import numpy as np

def calculate_lucas_kanade(prvs, next_frame, p0, lk_params):
    """
    Calculate optical flow using Lucas-Kanade method.
    """
    if p0 is not None:
        p1, status, err = cv2.calcOpticalFlowPyrLK(prvs, next_frame, p0, None, **lk_params)
        good_new, good_old = p1[status == 1], p0[status == 1]
        return good_new, good_old, status
    return None, None, None

def visualize_lucas_kanade(vis_frame, mask_lk, good_new, good_old, color):
    """
    Visualize Lucas-Kanade optical flow.
    """
    if good_new is not None and good_old is not None:
         for i, (new, old) in enumerate(zip(good_new, good_old)):
             a, b = new.ravel().astype(int)
             c, d = old.ravel().astype(int)
             mask_lk = cv2.line(mask_lk, (a, b), (c, d), color[i].tolist(), 2)
             vis_frame = cv2.circle(vis_frame, (a, b), 5, color[i].tolist(), -1)
         display_img = cv2.add(vis_frame, mask_lk)
         return display_img, good_new.reshape(-1, 1, 2), mask_lk
    return vis_frame, None, mask_lk