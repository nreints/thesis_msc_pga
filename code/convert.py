import torch
import numpy as np
from new_mujoco import fast_rotVecQuat
from pyquaternion import Quaternion
from rotation_Thijs import quat2mat

def eucl2pos(eucl_motion, start_pos):
    # print(eucl.shape, start_pos.shape)
    # NN
    # (128 x 12), (128 x 8 x 3)
    # LSTM
    # (128 x 20 x 12), (128 x 8 x 3)
    """
    Input:
        eucl_motion: Original predictions (euclidean motion)
        start_pos: Start position of simulation

    Output:
        Converted eucledian motion to current xyz position
    """
    # In case of fcnn
    if len(eucl_motion.shape) == 2:
        out = torch.empty_like(start_pos)
        for batch in range(out.shape[0]):
            out[batch] =  (eucl_motion[batch, :9].reshape(3,3) @ start_pos[batch].T + torch.vstack([eucl_motion[batch, 9:]]*8).T).T

    # In case of LSTM
    else:
        out = torch.empty((eucl_motion.shape[0], eucl_motion.shape[1], start_pos.shape[-2], start_pos.shape[-1]))

        n_frames = eucl_motion.shape[1]
        for batch in range(out.shape[0]):
            for frame in range(n_frames):
                # Reshape first 9 elements in rotation matrix and multiply with start_pos
                rotated_start = eucl_motion[batch, frame, :9].reshape(3,3) @ start_pos[batch].T

                # Add translation to each rotated_start
                out[batch, frame] =  (rotated_start.T + torch.vstack([eucl_motion[batch, frame, 9:]]*8))

    return out.reshape((out.shape[0], -1))


def quat2pos(quat, start_pos):
    # print(quat.shape, start_pos.shape)
    # NN
    # (128 x 7), (128 x 8 x 3)
    # LSTM
    # (128 x 20 x 7), (128 x 8 x 3)

    """
    Input:
        quat: Original predictions (quaternion motion)
            (batch, .., 7)
        start_pos: Start position of simulation
            (batch, .., 8, 3)

    Output:
        Converted quaternion to current xyz position
    """

    device = quat.device

    # In case of fcnn
    if len(quat.shape) == 2:

        batch, vert_num, dim = start_pos.shape

        out = torch.empty_like(start_pos).to(device)

        rotated_start = fast_rotVecQuat(start_pos, quat[:,:4])

        # X_start, Y_start, Z_start = start_pos[0][:, 0], start_pos[0][:, 1], start_pos[0][:, 2]
        # X_rot, Y_rot, Z_rot = rotated_start[0][:, 0], rotated_start[0][:, 1], rotated_start[0][:, 2]

        # distance_start = ((X_start[0] - X_start[1])**2 + (Y_start[0] - Y_start[1])**2 + (Z_start[0] - Z_start[1])**2)**0.5
        # distance_rot = ((X_rot[0] - X_rot[1])**2 + (Y_rot[0] - Y_rot[1])**2 + (Z_rot[0] - Z_rot[1])**2)**0.5

        # print(distance_start)
        # print(distance_rot)

        # print(quat[0][4:])
        # print()

        repeated_trans = quat[:, 4:][:, None, :].repeat(1,8,1)

        out = rotated_start + repeated_trans

        return out

    # In case of LSTM
    else:
        batch, vert_num, dim = start_pos.shape
        out = torch.empty((quat.shape[1], batch, vert_num, dim)).to(device)

        for frame in range(quat.shape[1]):
            rotated_start = fast_rotVecQuat(start_pos, quat[:, frame, :4])
            repeated_trans = quat[:, frame, 4:][:, None, :].repeat(1, 8, 1)
            out[frame] = (rotated_start + repeated_trans).reshape((batch, vert_num, dim))

        # Batch first
        out = torch.permute(out, (1, 0, 2, 3))
        return out



def log_quat2pos(log_quat, start_pos):
    # print(start_pos.shape, log_quat.shape)
    # NN
    # (128 x 7), (128 x 8 x 3)
    # LSTM
    # (128 x 20 x 7), (128 x 8 x 3)
    """
    Input:
        log_quat: Original predictions (log quaternion motion)
        start_pos: Start position of simulation

    Output:
        Converted log quaternion to current xyz position

        a bi cj dk = a vec{v}
    """
    # In case of fcnn
    if len(log_quat.shape) == 2:

        v = log_quat[:, 1:4]
        # print("V", v.shape)

        v_norm = torch.linalg.norm(v, dim=1)
        # print("v_norm", v_norm.shape)

        # Normalize v
        vec = torch.div(v.T, v_norm).T

        # v_norm = 0 --> div regel wordt NaN
        ################### Maybe correct#
        vec = torch.nan_to_num(vec)
        ######

        magn = torch.exp(log_quat[:, 0])
        # print("m", magn.shape)

        # --> sin(0) = 0 --> vector = torch.zeros if v_norm = 0
        # print("part1", torch.mul(magn, torch.sin(v_norm)).shape)
        vector = torch.mul(torch.mul(magn, torch.sin(v_norm)), vec.T).T
        # print("vector", vector.shape)

        scalar = (magn * torch.cos(v_norm))[:, None]

        quat = torch.hstack((scalar, vector))

        # print(Quaternion(np.array(quat[0])))

        # Stack translation to quaternion
        full_quat = torch.hstack((quat, log_quat[:, 4:]))
        # print("log", log_quat[0])
        # print("quat", full_quat[0])

        # print("start", start_pos[0])

        return quat2pos(full_quat, start_pos)

    # In case of LSTM
    else:
        v = log_quat[:, :, 1:4]
        v_norm = torch.linalg.norm(v, dim=2)

        # Normalize v
        vec = torch.div(v.permute((2, 0, 1)), v_norm).permute((1, 2, 0))

        magn = torch.exp(log_quat[:, :, 0])

        vector = torch.mul(torch.mul(magn, torch.sin(v_norm)), vec.permute((2, 0, 1))).permute((1, 2, 0))

        scalar = (magn * torch.cos(v_norm))[:, :, None]

        quat = torch.cat((scalar, vector), dim=2)

        # Stack translation to quaternion
        full_quat = torch.cat((quat, log_quat[:, :, 4:]), dim=2)

        return quat2pos(full_quat, start_pos)

def diff_pos_start2pos(true_preds, start_pos):
    """
    Input:
        true_preds: Original predictions (difference compared to start)
            Shape [batch_size, frames, datapoints]
        start_pos: Start position of simulation
            Shape [batch_size, datapoints]

    Output:
        Converted difference to current position

    """
    start_pos = start_pos.reshape(start_pos.shape[0], 8, 3)
    if len(true_preds.shape) == 2:
        true_preds = true_preds[:, None, :]
        start_pos = start_pos[:, None, :]

    start_pos = start_pos.reshape(-1, 1, true_preds.shape[2]).expand(-1, true_preds.shape[1], -1)
    # print(true_preds[0])
    result = start_pos + true_preds
    return result.reshape(result.shape[0], 8, 3)

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
    if data_type == "pos" or data_type == "pos_norm":
        return true_preds
        # return true_preds.reshape(true_preds.shape[0], 8, 3)
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
