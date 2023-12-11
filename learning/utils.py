import numpy as np
import cv2

def compute_rre(R_est: np.ndarray, R_gt: np.ndarray):
    """Compute the relative rotation error (geodesic distance of rotation)."""
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)
    # relative rotation error (RRE)
    rre = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt) - 1), -1.0, 1.0))
    return rre

def compute_rte(t_est: np.ndarray, t_gt: np.ndarray):
    assert t_est.shape == (3,), 't_est: expected shape (3,), received shape {}.'.format(t_est.shape)
    assert t_gt.shape == (3,), 't_gt: expected shape (3,), received shape {}.'.format(t_gt.shape)
    # relative translation error (RTE)
    rte = np.linalg.norm(t_est - t_gt)
    return rte


OBJ_NAMES_12 = ['a_lego_duplo', 'b_lego_duplo', 'bleach_cleanser', 'c_toy_airplane', 'cracker_box', 'd_toy_airplane', 'e_lego_duplo', 'e_toy_airplane', 'foam_brick', 'g_lego_duplo', 'gelatin_box', 'jenga', 'master_chef_can', 'mustard_bottle', 'nine_hole_peg_test', 'potted_meat_can', 'prism', 'pudding_box', 'rubiks_cube', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can', 'wood_block']

OBJ_NAMES_TO_IDX_12 = {'a_lego_duplo': 0, 'b_lego_duplo': 1, 'bleach_cleanser': 2, 'c_toy_airplane': 3, 'cracker_box': 4, 'd_toy_airplane': 5, 'e_lego_duplo': 6, 'e_toy_airplane': 7, 'foam_brick': 8, 'g_lego_duplo': 9, 'gelatin_box': 10, 'jenga': 11, 'master_chef_can': 12, 'mustard_bottle': 13, 'nine_hole_peg_test': 14, 'potted_meat_can': 15, 'prism': 16, 'pudding_box': 17, 'rubiks_cube': 18, 'sugar_box': 19, 'tomato_soup_can': 20, 'tuna_fish_can': 21, 'wood_block': 22}

IDX_TO_OBJ_NAMES_12 = dict((v,k) for k,v in OBJ_NAMES_TO_IDX_12.items())


OBJ_NAMES = OBJ_NAMES_12 + ['bowl_a', 'plate', 'pan_tefal', 'bowl']

OBJ_NAMES_TO_IDX = OBJ_NAMES_TO_IDX_12 | {'bowl_a': 23, 'plate': 24, 'pan_tefal': 25, 'bowl': 26}

IDX_TO_OBJ_NAMES = dict((v,k) for k,v in OBJ_NAMES_TO_IDX.items())

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x, y, w, h = tmp_x, tmp_y, tmp_w, tmp_h
    return [x, y, w, h]

def get_bbox(bbox):
    img_width = 720
    img_length = 1280
    step = 40
    border_list = [-1] + np.arange(step, img_width+step+step, step=step).tolist()

    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= img_width:
        bbx[1] = img_width-1
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= img_length:
        bbx[3] = img_length-1                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
