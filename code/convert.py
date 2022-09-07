import torch
import numpy as np
from new_mujoco import fast_rotVecQuat, own_rotVecQuat
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
    # print(start_pos.shape, eucl_motion.shape)
    if len(eucl_motion.shape) == 2:
        out = torch.empty_like(start_pos)
        for batch in range(out.shape[0]):
            out[batch] =  (eucl_motion[batch, :9].reshape(3,3) @ start_pos[batch].T + torch.vstack([eucl_motion[batch, 9:]]*8).T).T

        # eucl_motion = eucl_motion[:, None, :]
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


    if len(quat.shape) == 2:

        batch, vert_num, dim = start_pos.shape
        out = torch.empty_like(start_pos)

        rotated_start = fast_rotVecQuat(start_pos, quat[:,:4])
        repeated_trans = torch.repeat_interleave(quat[:, 4:], repeats=8, dim=0)
        out = rotated_start + repeated_trans
        return out.reshape((batch, vert_num, dim))

    else:
        batch, vert_num, dim = start_pos.shape
        out = torch.empty((quat.shape[1], batch, vert_num, dim))
        for frame in range(quat.shape[1]):
            rotated_start = fast_rotVecQuat(start_pos, quat[:,frame,:4])
            repeated_trans = torch.repeat_interleave(quat[:,frame,4:], repeats=8, dim=0)
            out[frame] = (rotated_start + repeated_trans).reshape((batch, vert_num, dim))
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
    """
    # print(len(log_quat.shape))
    if len(log_quat.shape) == 2:
        rot_vec = log_quat[:, :3]
        print(rot_vec)

        angle = log_quat[:, 3]
        print(angle)
        trans = log_quat[:, 4:]
        print(trans)

        cos = torch.cos(angle/2).reshape(-1, 1)
        sin = torch.sin(angle/2)


        quat = torch.empty(log_quat.shape)
        quat[:, 0] = cos.squeeze()
        part1_1 = rot_vec * torch.vstack([sin]*3).T
        quat[:, 1:4] = part1_1
        quat[:, 4:] = trans
        print("dhwie",quat)
        exit()

        return quat2pos(quat, start_pos)

    else:
        rot_vec = log_quat[:, :, :3]
        angle = log_quat[:, :, 3]
        trans = log_quat[:, :, 4:]

        cos = torch.cos(angle/2)
        sin = torch.sin(angle/2)

        quat = torch.empty(log_quat.shape)
        quat[:, :, 0] = cos
        quat[:, :, 1:4] = rot_vec * sin.unsqueeze(2).repeat(1, 1, 3)
        quat[:, :, 4:] = trans

        return quat2pos(quat, start_pos)


# start = [[[1,0,0],[2,0,0],[0.5,0,0]],
#             [[0,1,0],[0,2,0],[0,0.5,0]]]

# start_nn = torch.tensor([[[1,0,0],[2,0,0],[0.5,0,0]]])

# log_quat = [[[1,0,0.7,0.4,0,0,-0.2]],
#             [[0.3,0,0.7,0.4,0,0,-0.3]]]
# log_quat_nn = [[[1,0,0.7,0.4,0,0,-0.2]]]

# lq = torch.tensor([[0.304, 0.397 , 0.616, 0.609]])
# # q =  Quaternion.random()
# # print(q)
# # print(Quaternion.exp(q))

# print(log_quat2pos(lq, start_nn))


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
    if len(true_preds.shape) == 2:
        true_preds = true_preds[:, None, :]
        start_pos = start_pos[:, None, :]

    start_pos = start_pos.reshape(-1, 1, true_preds.shape[2]).expand(-1, true_preds.shape[1], -1)
    # start_pos = start_pos.astype('float64')
    return start_pos + true_preds



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
