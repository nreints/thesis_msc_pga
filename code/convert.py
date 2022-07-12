import torch
import numpy as np
from new_mujoco import rotVecQuat

def eucl2pos(eucl_motion, start_pos):
    """

    Input:
        eucl_motion: Original predictions (euclidean motion)
        start_pos: Start position of simulation

    Output:
        Converted eucledian motion to current position
    """
    out = np.empty_like(start_pos)
    eucl_motion = eucl_motion.numpy().astype('float64')
    start_pos = start_pos.numpy().astype('float64')
    for batch in range(out.shape[0]):
        out[batch] =  (eucl_motion[batch,:9].reshape(3,3) @ start_pos[batch].T + np.vstack([eucl_motion[batch, 9:]]*8).T).T

    return torch.from_numpy(out.reshape((out.shape[0], -1)))


def quat2pos(quat, start_pos):
    """

    Input:
        quat: Original predictions (quaternion motion)
        start_pos: Start position of simulation

    Output:
        Converted quaternion to current position
    """
    out = np.empty_like(start_pos)

    if not isinstance(quat, np.ndarray):
        quat = quat.numpy().astype('float64')
    if not isinstance(start_pos, np.ndarray):
        start_pos = start_pos.numpy().astype('float64')
    for batch in range(out.shape[0]):
        for vert in range(out.shape[1]):
            out[batch, vert] = rotVecQuat(start_pos[batch,vert,:], quat[batch, :4]) + quat[batch, 4:]

    return torch.from_numpy(out.reshape((out.shape[0], -1)))

def log_quat2pos(log_quat, start_pos):
    """

    Input:
        log_quat: Original predictions (log quaternion motion)
        start_pos: Start position of simulation

    Output:
        Converted log quaternion to current position
    """
    log_quat = log_quat.numpy().astype('float64')
    start_pos = start_pos.numpy().astype('float64')
    rot_vec = log_quat[:, :3]
    angle = log_quat[:,3]
    trans = log_quat[:,4:]
    cos = np.cos(angle/2).reshape(-1, 1)
    sin = np.sin(angle/2)
    part1_1 = rot_vec * np.vstack([sin]*3).T
    part1 = np.append(cos, part1_1, axis=1)
    quat = np.append(part1, trans, axis=1)

    return quat2pos(quat, start_pos)

def diff_pos_start2pos(true_preds, start_pos):
    """

    Input:
        true_preds: Original predictions (difference compared to start)
        start_pos: Start position of simulation

    Output:
        Converted difference to current position
    """
    start_pos = start_pos.reshape(-1, 1, true_preds.shape[2]).expand(-1, true_preds.shape[1], -1)
    start_pos = start_pos.numpy().astype('float64')
    return torch.from_numpy(start_pos + true_preds.numpy().astype('float64'))

def convert(true_preds, start_pos, data_type):
    """
    Converts true predictions given data type.
    Input:
        true_preds: Prediction of model
        start_pos: Start position of simulation
        data_type: Datatype of predictions

    Output:
        Converted true predictions.

    """
    if data_type == "pos":
        return true_preds
    elif data_type == "eucl_motion":
        return eucl2pos(true_preds, start_pos)
    elif data_type == "quat":
        return quat2pos(true_preds, start_pos)
    elif data_type == "log_quat":
        return log_quat2pos(true_preds, start_pos)
    # elif data_type == "pos_diff":
    #     return diff_pos2pos(true_preds, start_pos)
    elif data_type == "pos_diff_start":
        return diff_pos_start2pos(true_preds, start_pos)
    return True
