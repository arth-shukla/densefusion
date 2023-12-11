import numpy as np
import scipy
from tqdm import tqdm

def R_and_T(T):
    return T[:3, :3], T[:3, 3]

def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def svd_rot(M):
    U, _, VT = np.linalg.svd(M)
    V = VT.T
    R = U @ V.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = U @ V.T
    return R

def rigid_transform(ps, qs):
    
    pmean = np.mean(ps, axis=0)
    qmean = np.mean(qs, axis=0)
    pbars = ps - pmean
    qbars = qs - qmean

    # project onto space of orthogonal matrices
    M = qbars.T @ pbars
    R = svd_rot(M)
    t = qmean - R @ pmean

    return to_T(R, t)

def sample_3d_point(r=1):
    rho = r * (1 + (np.random.rand() / 2))
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * np.pi
    return np.array([
        rho * np.cos(theta) * np.sin(phi),
        rho * np.sin(theta) * np.sin(phi),
        rho * np.cos(phi),
    ])

def run_icp(source_pcd, target_pcd, max_attempts=10, max_iters=1000, finish_loop_thresh=1e-5, acceptable_thresh=1e-5, pbar=False, seed=0):
    """Iterative closest point.
    
    Args:
        source_pcd (np.ndarray): [N1, 3]
        target_pcd (np.ndarray): [N2, 3]
    
    Returns:
        np.ndarray: [4, 4] rigid transformation to align source to target.
    """

    np.random.seed(seed)

    targ_center = np.mean(target_pcd, axis=0)
    targ_rad = np.max(np.linalg.norm(target_pcd - targ_center, axis=1))

    attempt_dists = []
    attempt_Ts = []

    attempt_iter = tqdm(range(max_attempts)) if pbar else range(max_attempts)
    for attempt in attempt_iter:

        ps = source_pcd 
        # for retries:
        # (1) move to point on 4*targ_rad ball around targ pcd
        # (2) randomly rotate
        if attempt > 0:
            ps = ps - np.mean(ps, axis=0) + targ_center
            ps = ps + sample_3d_point(r = 2 * targ_rad)
            ps = ps @ svd_rot(np.random.rand(3, 3))

        last_dist = np.inf

        tree = scipy.spatial.KDTree(target_pcd)
        qs = target_pcd[tree.query(ps)[1]]
        for iter_num in range(max_iters):

            # get transformation
            T = rigid_transform(ps, qs)

            # transform pt cld
            ps = ps @ T[:3, :3].T + T[:3, 3]

            # update qs
            qs = target_pcd[tree.query(ps)[1]]

            # if avg_dist doesn't change, ret early, else keep going
            avg_dist = np.mean(np.linalg.norm(ps - qs, axis=1))
            if np.abs(avg_dist - last_dist) <= finish_loop_thresh:
                break
            last_dist = avg_dist
        
        # if error acceptably small, continue
        T = rigid_transform(source_pcd, ps)
        if avg_dist <= acceptable_thresh:
            return T
            
        # else, we save all Ts and ret whichever is best
        attempt_dists.append(avg_dist)
        attempt_Ts.append(T)

    # ret transformation
    return attempt_Ts[np.argmin(attempt_dists)]