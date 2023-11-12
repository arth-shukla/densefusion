import numpy as np

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


OBJ_NAMES = ['a_lego_duplo', 'b_lego_duplo', 'bleach_cleanser', 'c_toy_airplane', 'cracker_box', 'd_toy_airplane', 'e_lego_duplo', 'e_toy_airplane', 'foam_brick', 'g_lego_duplo', 'gelatin_box', 'jenga', 'master_chef_can', 'mustard_bottle', 'nine_hole_peg_test', 'potted_meat_can', 'prism', 'pudding_box', 'rubiks_cube', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can', 'wood_block']

OBJ_NAMES_TO_IDX = {'a_lego_duplo': 0, 'b_lego_duplo': 1, 'bleach_cleanser': 2, 'c_toy_airplane': 3, 'cracker_box': 4, 'd_toy_airplane': 5, 'e_lego_duplo': 6, 'e_toy_airplane': 7, 'foam_brick': 8, 'g_lego_duplo': 9, 'gelatin_box': 10, 'jenga': 11, 'master_chef_can': 12, 'mustard_bottle': 13, 'nine_hole_peg_test': 14, 'potted_meat_can': 15, 'prism': 16, 'pudding_box': 17, 'rubiks_cube': 18, 'sugar_box': 19, 'tomato_soup_can': 20, 'tuna_fish_can': 21, 'wood_block': 22}

IDX_TO_OBJ_NAMES = dict((v,k) for k,v in OBJ_NAMES_TO_IDX.items())