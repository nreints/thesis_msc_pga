import torch
import numpy as np
from new_mujoco import own_rotVecQuat

def eucl2pos(eucl_motion, start_pos):
    """
    Input:
        eucl_motion: Original predictions (euclidean motion)
        start_pos: Start position of simulation

    Output:
        Converted eucledian motion to current position
    """
    # Convert to [batch, 1, ...]
    if len(eucl_motion.shape) == 2:
        out = torch.empty_like(start_pos)
        # print(eucl_motion.dtype)
        # eucl_motion = eucl_motion.astype('float64')
        # start_pos = start_pos.astype('float64')
        for batch in range(out.shape[0]):
            out[batch] =  (eucl_motion[batch,:9].reshape(3,3) @ start_pos[batch].T + torch.vstack([eucl_motion[batch, 9:]]*8).T).T

        # eucl_motion = eucl_motion[:, None, :]
        # start_pos = start_pos[:, None, :]
    else:
        # print(eucl_motion.shape, start_pos.shape)
        out = torch.empty((eucl_motion.shape[0], eucl_motion.shape[1], start_pos.shape[-1]))
        eucl_motion = eucl_motion.astype('float64')
        start_pos = start_pos.astype('float64')
        frames = eucl_motion.shape[1]
        # print(start_pos[0].shape)
        
        for batch in range(out.shape[0]):
            # out[batch] =  (eucl_motion[batch,:9].reshape(3,3) @ start_pos[batch].T + np.vstack([eucl_motion[batch, 9:]]*8).T).T

            for frame in range(frames):
                # print(print(out[batch, frame].shape))
                # print("reshaped_rot_mat", eucl_motion[batch, frame,:9].reshape(3,3).shape)
                # print("reshaped start pos", start_pos[batch].reshape(8,3).T.shape)
                # print("translation?", np.vstack([eucl_motion[batch, frame, 9:]]*8).T.shape)
                # print((eucl_motion[batch, frame,:9].reshape(3,3) @ start_pos[batch].reshape(8,3).T + np.vstack([eucl_motion[batch, frame, 9:]]*8).T).T.shape)

                rotated_start = eucl_motion[batch, frame, :9].reshape(3,3) @ start_pos[batch].reshape(8,3).T
                # print("rot_start", rotated_start.shape)
                # print((rotated_start + np.vstack([eucl_motion[batch, frame, 9:]]*8).T).flatten().shape)
                out[batch, frame] =  (rotated_start + np.vstack([eucl_motion[batch, frame, 9:]]*8).T).flatten()

            # start_posc = start_pos[batch].reshape((8,3))
            # reshaped_rot = eucl_motion[batch,:,:9].reshape(frames,3,3).reshape(60,-1)
            # print("res_rot", reshaped_rot.reshape(60,-1).shape)

            # print((reshaped_rot.squeeze() @ start_posc.squeeze().T).shape)

            # out[batch,:] =  (reshaped_rot.squeeze() @ start_posc.squeeze().T + np.vstack([eucl_motion[batch, :, 9:]]*8).T).T


    return out.reshape((out.shape[0], -1))


def quat2pos(quat, start_pos):
    """

    Input:
        quat: Original predictions (quaternion motion)
        start_pos: Start position of simulation

    Output:
        Converted quaternion to current position
    """
    out = torch.empty_like(start_pos)

    # if not isinstance(quat, np.ndarray):
    #     quat = quat.astype('float64')
    # if not isinstance(start_pos, np.ndarray):
    #     start_pos = start_pos.astype('float64')
    for batch in range(out.shape[0]):
        for vert in range(out.shape[1]):
            out[batch, vert] = own_rotVecQuat(start_pos[batch,vert,:], quat[batch, :4]) + quat[batch, 4:]

    return out.reshape((out.shape[0], -1))

def log_quat2pos(log_quat, start_pos):
    """

    Input:
        log_quat: Original predictions (log quaternion motion)
        start_pos: Start position of simulation

    Output:
        Converted log quaternion to current position
    """
    # log_quat = log_quat.astype('float64')
    # start_pos = start_pos.astype('float64')
    rot_vec = log_quat[:, :3]
    angle = log_quat[:,3]
    trans = log_quat[:,4:]
    cos = torch.cos(angle/2).reshape(-1, 1)
    sin = torch.sin(angle/2)


    quat = torch.empty(log_quat.shape)
    quat[:, 0] = cos.squeeze()
    part1_1 = rot_vec * torch.vstack([sin]*3).T
    quat[:, 1:4] = part1_1

    return quat2pos(quat, start_pos)

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
    start_pos = start_pos.astype('float64')
    return torch.from_numpy(start_pos + true_preds.astype('float64'))



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
