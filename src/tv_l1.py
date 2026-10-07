import cv2

def create_tvl1_algorithm():
    """
    Create Dual TV-L1 optical flow algorithm object.
    Requires opencv-contrib-python.
    """
    return cv2.optflow.createOptFlow_DualTVL1()

def calculate_tvl1(tvl1_algo, prvs, next_frame):
    """
    Calculate optical flow using TV-L1 method.
    """
    flow = tvl1_algo.calc(prvs, next_frame, None)
    return flow