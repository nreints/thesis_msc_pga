import torch
import roma
import time


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

        # TODO Werkt niet

        rotations = eucl_motion[:, :9].reshape(-1, 3, 3)
        mult = torch.bmm(rotations, start_pos.reshape(-1, 8, 3).mT).mT
        # print(mult.shape)
        # print(eucl_motion[:, 9:][:, None, :].flatten(start_dim=1).shape)
        out = (mult + eucl_motion[:, 9:][:, None, :]).flatten(start_dim=1)

        # out = torch.empty_like(start_pos)
        # for batch in range(out.shape[0]):
        #     # EINSUM # bij,bkj->bik
        #     out[batch] = (
        #         eucl_motion[batch, :9].reshape(3, 3) @ start_pos[batch].reshape(8,3).T
        #         + torch.vstack([eucl_motion[batch, 9:]] * 8).T
        #     ).T.flatten()

        return out

    # In case of LSTM
    else:

        # TODO TODO TODO Maak sneller
        rotations = eucl_motion[..., :9].reshape(eucl_motion.shape[0], eucl_motion.shape[1], 3, 3)
        flat_rotations = rotations.flatten(end_dim=1)


        #TODO TODO FIX in plot_data
        if len(start_pos.shape) != 3:
            start_pos = start_pos[:, None, :]
        correct_start_pos = start_pos.repeat(1, eucl_motion.shape[1], 1).flatten(end_dim=1)
        mult = torch.bmm(flat_rotations, correct_start_pos.reshape(-1, 8, 3).mT).mT

        out = (mult + eucl_motion.flatten(end_dim=1)[:, 9:][:, None, :]).flatten(start_dim=1)
        out = out.reshape(eucl_motion.shape[0], eucl_motion.shape[1], out.shape[-1])

        # Code above 24x as fast
        # out2 = torch.empty(
        #     (
        #         eucl_motion.shape[0],
        #         eucl_motion.shape[1],
        #         start_pos.shape[-1],
        #     )
        # , device=eucl_motion.device)
        # n_frames = eucl_motion.shape[1]
        # for batch in range(out2.shape[0]):
        #     for frame in range(n_frames):
        #         # Reshape first 9 elements in rotation matrix and multiply with start_pos
        #         rotated_start = (eucl_motion[batch, frame, :9].reshape(3, 3) @ start_pos[batch].reshape(8,3).T).T

        #         # Add translation to each rotated_start
        #         out2[batch, frame] = (rotated_start + eucl_motion[batch, frame, 9:][None, :]).flatten()

        # print(out - out2)

        return out


def fast_rotVecQuat(v, q):
    """
    Input:
        v: vector to be rotated
            shape: (* x 24)
        q: quaternion to rotate v
            shape: (* x 4)
    Output:
        Rotated batch of vectors v by a batch quaternion q.
    """
    device = v.device

    q_norm = torch.div(q.T, torch.norm(q, dim=-1)).T

    # Swap columns for roma calculations (bi, cj, dk, a)
    q_new1 = torch.index_select(q_norm, 1, torch.tensor([1, 2, 3, 0], device=device))

    # start = time.time()
    vT = v.mT
    # print("matT", time.time() - start)

    # TODO einsum 10x slower
    # start = time.time()
    # v_test = torch.einsum("...ij->...ji", v)
    # print("einsum", time.time() - start)
    # print(roma.unitquat_to_rotmat(q_new1).shape)
    # print(v.reshape(-1, 8,3).mT.shape)
    # print("poep", torch.bmm(roma.unitquat_to_rotmat(q_new1), v.reshape(-1, 8,3).mT).mT.shape)
    rot_mat = torch.bmm(roma.unitquat_to_rotmat(q_new1), v.reshape(-1, 8,3).mT).mT.to(device)

    return rot_mat


def quat2pos(quat, start_pos):
    # print(quat.shape, start_pos.shape)
    # NN
    # (128 x 7), (128 x 8 x 3)
    # LSTM
    # (128 x 20 x 7), (128 x 20 x 8 x 3)

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

        # batch, vert_num, dim = start_pos.shape
        # print(start_pos.shape)
        out = torch.empty_like(start_pos, device=device)

        rotated_start = fast_rotVecQuat(start_pos, quat[:, :4])

        repeated_trans = quat[:, 4:][:, None, :]

        out = rotated_start + repeated_trans
        return out.flatten(start_dim=1)

    # In case of LSTM
    else:
        quat_flat = quat.flatten(end_dim=1)
        if len(start_pos.shape) != 3:
            start_pos = start_pos[:, None, :]

        correct_start_pos = start_pos.repeat(1, quat.shape[1], 1).flatten(end_dim=1)
        # Rotate start by quaternion
        rotated_start = fast_rotVecQuat(correct_start_pos, quat_flat[:, :4])

        # Add Translation
        out = (rotated_start + quat_flat[:, 4:][:, None, :]).flatten(start_dim=1)
        # Fix shape
        out = out.reshape(quat.shape[0], quat.shape[1], out.shape[-1])

        ######################### OLD WAY 2x slower
        # vert_dim = start_pos.shape[-1]
        # batch = quat.shape[0]
        # n_frames = quat.shape[1]

        # out2 = torch.empty((n_frames, batch, vert_dim), device=device)

        # for frame in range(n_frames):
        #     rotated_start1 = fast_rotVecQuat(start_pos, quat[:, frame, :4])
        #     repeated_trans = quat[:, frame, 4:][:, None, :]
        #     out2[frame] = (rotated_start1 + repeated_trans).reshape(
        #         (batch, vert_dim)
        #     )


        # # # Batch first
        # out2 = torch.permute(out2, (1, 0, 2))

        # # print(out.shape, out2.shape)
        # print(out-out2)
        # exit()
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
            Shape:
        start_pos: Start position of simulation
            Shape:
    Output:
        Converted log quaternion to current xyz position

        a bi cj dk = a vec{v}
    """
    # In case of fcnn
    if len(log_quat.shape) == 2:

        v = log_quat[:, 1:4]

        v_norm = torch.linalg.norm(v, dim=1)

        # Normalize v
        vec = torch.div(v.T, v_norm).T

        # v_norm = 0 --> div regel wordt NaN
        vec = torch.nan_to_num(vec)

        magn = torch.exp(log_quat[:, 0])

        # --> sin(0) = 0 --> vector = torch.zeros if v_norm = 0
        vector = torch.mul(torch.mul(magn, torch.sin(v_norm)), vec.T).T

        scalar = (magn * torch.cos(v_norm))[:, None]

        quat = torch.hstack((scalar, vector))

        # Stack translation to quaternion
        full_quat = torch.hstack((quat, log_quat[:, 4:]))

        return quat2pos(full_quat, start_pos)

    # In case of LSTM
    else:
        v = log_quat[:, :, 1:4]
        v_norm = torch.linalg.norm(v, dim=2)

        # Normalize v
        vec = torch.div(v.permute((2, 0, 1)), v_norm).permute((1, 2, 0))

        magn = torch.exp(log_quat[:, :, 0])

        vector = torch.mul(
            torch.mul(magn, torch.sin(v_norm)), vec.permute((2, 0, 1))
        ).permute((1, 2, 0))

        scalar = (magn * torch.cos(v_norm))[:, :, None]

        quat = torch.cat((scalar, vector), dim=2)

        # Stack translation to quaternion
        full_quat = torch.cat((quat, log_quat[:, :, 4:]), dim=2)

        return quat2pos(full_quat, start_pos)



def dualQ2pos(dualQ, start_pos):
    """
    Input:
        dualQ: Original predictions (Dual quaternion)
            Shape (batch_size x * x 8)
        start_pos: Start position of simulation
            Shape (batch_size x * x 8 x 3)

            (* is only present for lstm)

    Output:
        Converted Dual-quaternion to current position
    """
    device = dualQ.device


    qr_dim = dualQ[..., :4].shape

    qr = dualQ[..., :4].flatten(0, -2)
    qd = dualQ[..., 4:].flatten(0, -2)


    swapped_ind = torch.tensor([1, 2, 3, 0], device=device)
    qr_roma = torch.index_select(qr, 1, swapped_ind)
    conj_qr = roma.quat_conjugation(qr_roma)

    qd_roma = torch.index_select(qd, 1, swapped_ind)

    t = 2 * roma.quat_product(qd_roma, conj_qr)

    qr = qr.reshape(qr_dim)
    t = t.reshape(qr_dim)

    # Concatenate and delete zeros column
    quaternion = torch.cat((qr, t[..., :-1]), dim=-1)

    converted_pos = quat2pos(quaternion, start_pos)

    return converted_pos

def log_dualQ2pos(logDualQ_in, start_pos):
    """
    Input bivector (6 numbers) returns rotor (=exp of bivector) (8 numbers)
    (17 mul, 8 add, 2 div, 1 sincos, 1 sqrt)
    """
    # print(logDualQ.flatten(start_dim=0, end_dim=-2).shape)
    out_shape = list(logDualQ_in.shape)
    out_shape[-1] = 8

    # 128, 6
    # 128, 30, 6
    logDualQ = logDualQ_in.flatten(start_dim=0, end_dim=-2)
    l = logDualQ[:, 3] * logDualQ[:, 3] + logDualQ[:, 4] * logDualQ[:, 4] + logDualQ[:, 5] * logDualQ[:, 5]
    mask = (l == 0)[:, None]
    ones = torch.ones_like(l)
    zeros = torch.zeros_like(l)
    alternative = torch.stack([ones, zeros, zeros, zeros, zeros, -logDualQ[:, 0], -logDualQ[:, 1], -logDualQ[:, 2]]).T
    m = logDualQ[:, 0] * logDualQ[:, 5] + logDualQ[:, 1] * logDualQ[:, 4] + logDualQ[:, 2] * logDualQ[:, 3]
    a = torch.sqrt(l)
    c = torch.cos(a)
    s = torch.sin(a) / a
    t = m / l * (c - s)
    dualQ = torch.stack([
                        c,
                        s * logDualQ[:, 5],
                        s * logDualQ[:, 4],
                        s * logDualQ[:, 3],
                        m * s,
                        -s * logDualQ[:, 0] - t * logDualQ[:, 5],
                        -s * logDualQ[:, 1] - t * logDualQ[:, 4],
                        -s * logDualQ[:, 2] - t * logDualQ[:, 3]
                    ]).T
    dualQ = mask * alternative + (~mask) * dualQ
    # 128, 8
    # 128, 30, 8
    return dualQ2pos(dualQ.reshape(out_shape), start_pos)


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
    # start_pos = start_pos.reshape(start_pos.shape[0], 8, 3)

    if len(true_preds.shape) == 2:
        true_preds = true_preds[:, None, :]
    if len(start_pos.shape) == 2:
        start_pos = start_pos[:, None, :]

    result = start_pos + true_preds
    return result.squeeze()


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
    # print("prediction", true_preds.shape)
    # print("start_pos", start_pos.shape)
    # exit()
    if data_type == "pos" or data_type == "pos_norm":
        return true_preds
    elif data_type == "eucl_motion":
        return eucl2pos(true_preds, start_pos)
    elif data_type == "quat":
        return quat2pos(true_preds, start_pos)
    elif data_type == "log_quat":
        return log_quat2pos(true_preds, start_pos)
    elif data_type == "dual_quat":
        return dualQ2pos(true_preds, start_pos)
    # elif data_type == "pos_diff":
    #     return diff_pos2pos(true_preds, start_pos)
    elif data_type == "pos_diff_start":
        return diff_pos_start2pos(true_preds, start_pos)
    elif data_type =="log_dualQ":
        return log_dualQ2pos(true_preds, start_pos)
    raise Exception(f"{data_type} cannot be converted")
