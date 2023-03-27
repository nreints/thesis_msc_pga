import torch
import roma

# import time


def eucl2pos(eucl_motion, start_pos, xpos_start):
    if len(eucl_motion.shape) == 2:
        if xpos_start is None:
            print("print correct")
            xpos_start = 0
        else:
            xpos_start = xpos_start.reshape(-1, 1, 3)
        if torch.any(torch.isnan(eucl_motion)):
            print("FUCK")
            exit()
        rotations = eucl_motion[:, :9].reshape(-1, 3, 3)
        if torch.any(torch.isnan(rotations)):
            print("1")
            exit()
        # print("Convert", rotations)
        start_origin = (start_pos.reshape(-1, 8, 3) - xpos_start).mT
        if torch.any(torch.isnan(start_origin)):
            print("2")
            exit()
        mult = torch.bmm(rotations, start_origin).mT
        if torch.any(torch.isnan(mult)):
            print("3")
            exit()
        out = mult + eucl_motion[:, 9:].reshape(-1, 1, 3) + xpos_start
        if torch.any(torch.isnan(out)):
            print("4")
            exit()
        return out.flatten(start_dim=1)
    # In case of LSTM/GRU
    else:
        print("APPEL        MOES NOG FIXEN")
        rotations = eucl_motion[..., :9].reshape(
            eucl_motion.shape[0], eucl_motion.shape[1], 3, 3
        )
        flat_rotations = rotations.flatten(end_dim=1)

        correct_start_pos = start_pos.repeat(1, eucl_motion.shape[1], 1).flatten(
            end_dim=1
        )
        mult = torch.bmm(flat_rotations, correct_start_pos.reshape(-1, 8, 3).mT).mT

        out = (mult + eucl_motion.flatten(end_dim=1)[:, 9:][:, None, :]).flatten(
            start_dim=1
        )
        out = out.reshape(eucl_motion.shape[0], eucl_motion.shape[1], out.shape[-1])
        return out


def eucl2pos_ori(eucl_motion, start_pos):
    """
    Transforms a batch of vectors by a rotation matrix and translation vector.

    Input:
        eucl_motion: Original predictions (euclidean motion)
            (batch x 12)
            (batch x frames x 20)
        start_pos: Start position of simulation
            (batch x 24)
            (batch x 24)

    Output:
        Converted eucledian motion to current xyz position
            (batch x 24)
            (batch x frames x 24)
    """
    # In case of fcnn
    if len(eucl_motion.shape) == 2:
        rotations = eucl_motion[:, :9].reshape(-1, 3, 3)
        mult = torch.bmm(rotations, start_pos.reshape(-1, 8, 3).mT).mT

        out = (mult + eucl_motion[:, 9:][:, None, :]).flatten(start_dim=1)
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

    rotated_v = torch.bmm(
        roma.unitquat_to_rotmat(q_new1), v.reshape(-1, 8, 3).mT
    ).mT.to(device)

    return rotated_v


def quat2pos(quat, start_pos, xpos_start):
    if len(quat.shape) == 2:
        if xpos_start is None:
            xpos_start = 0
        else:
            xpos_start = xpos_start.reshape(-1, 1, 3)
        start_pos_shape = start_pos.shape
        start_origin = (start_pos.reshape(-1, 8, 3) - xpos_start).reshape(
            start_pos_shape
        )
        rotated_start = fast_rotVecQuat(
            start_origin,
            quat[:, :4],
        )
        # print(quat[:, 4:])
        repeated_trans = quat[:, 4:][:, None, :]
        # print(repeated_trans)

        # if torch.all(quat[:, 4:] == 0):
        #     print("all zeros!!")
        # if torch.all(repeated_trans[:,:,4:] == 0):
        #     print("repeated zeros!!")
        out = rotated_start + repeated_trans + xpos_start
        return out.flatten(start_dim=1)


def quat2pos_ori(quat, start_pos):
    """
    Input:
        - quat: Original predictions (quaternion motion)
            (batch, 7) or (batch, frames, 7)
        - start_pos: Start position of simulation
            (batch, 24) or (batch, frames, 24)

    Output:
        - Converted quaternion to current xyz position
            (batch, 24) or (batch, frames, 24)
    """
    # In case of fcnn
    if len(quat.shape) == 2:
        rotated_start = fast_rotVecQuat(start_pos, quat[:, :4])
        repeated_trans = quat[:, 4:][:, None, :]
        out = (rotated_start + repeated_trans).flatten(start_dim=1)
        return out

    # In case of LSTM
    else:
        quat_flat = quat.flatten(end_dim=1)
        # For visualisation
        if len(start_pos.shape) != 3:
            start_pos = start_pos[:, None, :]

        repeated_start_pos = start_pos.repeat(1, quat.shape[1], 1).flatten(end_dim=1)

        # Rotate start by quaternion
        rotated_start = fast_rotVecQuat(repeated_start_pos, quat_flat[:, :4])

        # Add Translation
        out = (rotated_start + quat_flat[:, 4:][:, None, :]).flatten(start_dim=1)
        # Fix shape
        out = out.reshape(quat.shape[0], quat.shape[1], out.shape[-1])
        return out


def log_quat2pos(log_quat, start_pos, start_xpos):
    """
    Input:
        - log_quat: Original predictions (log quaternion motion)
            Shape: (batch, 7) or (batch, frames, 7)
        - start_pos: Start position of simulation
            Shape: (batch, 24) or (batch, frames, 24)

    Output:
        - Converted log quaternion to current xyz position
            (batch, 24) or
            (batch, frames, 24)
        a bi cj dk = a vec{v}
    """
    # In case of fcnn
    if len(log_quat.shape) == 2:
        # Select the second part of the quaternion.
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

        return quat2pos(full_quat, start_pos, start_xpos)

    # In case of LSTM /GRU
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

        return quat2pos(full_quat, start_pos, start_xpos)


def dualQ2pos(dualQ, start_pos, start_xpos):
    """
    Input:
        - dualQ: Original predictions (Dual quaternion)
            Shape (batch_size, *, 8)
        - start_pos: Start position of simulation
            Shape (batch_size, *, 8, 3)

    Output:
        - Converted Dual-quaternion to current position
            (batch x 24) or
            (batch x frames x 24)
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

    converted_pos = quat2pos(quaternion, start_pos, start_xpos)
    return converted_pos


def log_dualQ2pos(logDualQ_in, start_pos, start_xpos):
    """
    Input bivector (6 numbers) returns position by first calculating the dual quaternion = exp(log_dualQ).
    (17 mul, 8 add, 2 div, 1 sincos, 1 sqrt)
    """
    """
    Input:
        - log_dualQ: Original predictions (logarithm of dual quaternion)
            Shape (batch_size, 6) or (batch_size, frames, 6)
        - start_pos: Start position of simulation
            Shape (batch_size, 24) or (batch_size, frames, 24)

    Output:
        - Converted Dual-quaternion to current position
            (batch, 24) or
            (batch, frames, 24)
    """
    out_shape = list(logDualQ_in.shape)
    out_shape[-1] = 8

    logDualQ = logDualQ_in.flatten(start_dim=0, end_dim=-2)
    l = (
        logDualQ[:, 3] * logDualQ[:, 3]
        + logDualQ[:, 4] * logDualQ[:, 4]
        + logDualQ[:, 5] * logDualQ[:, 5]
    )
    mask = (l == 0)[:, None]
    ones_tensor = torch.ones_like(l)
    zeros_tensor = torch.zeros_like(l)
    alternative = torch.stack(
        [
            ones_tensor,
            zeros_tensor,
            zeros_tensor,
            zeros_tensor,
            zeros_tensor,
            -logDualQ[:, 0],
            -logDualQ[:, 1],
            -logDualQ[:, 2],
        ]
    ).T
    m = (
        logDualQ[:, 0] * logDualQ[:, 5]
        + logDualQ[:, 1] * logDualQ[:, 4]
        + logDualQ[:, 2] * logDualQ[:, 3]
    )
    a = torch.sqrt(l)
    c = torch.cos(a)
    s = torch.sin(a) / a
    t = m / l * (c - s)
    dualQ = torch.stack(
        [
            c,
            s * logDualQ[:, 5],
            s * logDualQ[:, 4],
            s * logDualQ[:, 3],
            m * s,
            -s * logDualQ[:, 0] - t * logDualQ[:, 5],
            -s * logDualQ[:, 1] - t * logDualQ[:, 4],
            -s * logDualQ[:, 2] - t * logDualQ[:, 3],
        ]
    ).T
    dualQ = mask * alternative + (~mask) * dualQ

    return dualQ2pos(dualQ.reshape(out_shape), start_pos, start_xpos)


def diff_pos_start2pos(true_preds, start_pos):
    """
    Input:
        - true_preds: Original predictions (difference compared to start)
            Shape [batch_size, frames, datapoints]
        - start_pos: Start position of simulation
            Shape [batch_size, datapoints]
    Output:
        - Converted difference to current position
    """

    if len(true_preds.shape) == 2:
        true_preds = true_preds[:, None, :]
    if len(start_pos.shape) == 2:
        start_pos = start_pos[:, None, :]

    result = start_pos + true_preds
    return result.squeeze()


def convert(true_preds, start_pos, data_type, xpos_start=None):
    """
    Converts true predictions given data type.
    Input:
        - true_preds: Prediction of model
        - start_pos: Start position of simulation
        - data_type: Datatype of predictions
    Output:
        - Converted true predictions.
    """
    if data_type == "pos" or data_type == "pos_norm":
        return true_preds
    elif data_type == "eucl_motion" or (data_type[:-4] == "eucl_motion"):
        return eucl2pos(true_preds, start_pos, xpos_start)
    elif data_type == "quat":
        return quat2pos(true_preds, start_pos, xpos_start)
    elif data_type == "quat_ori":
        return quat2pos(true_preds, start_pos, xpos_start)
    elif data_type == "log_quat":
        return log_quat2pos(true_preds, start_pos, xpos_start)
    elif data_type == "dual_quat":
        return dualQ2pos(true_preds, start_pos, xpos_start)
    elif data_type == "pos_diff_start":
        return diff_pos_start2pos(true_preds, start_pos)
    elif data_type == "log_dualQ":
        return log_dualQ2pos(true_preds, start_pos, xpos_start)
    raise Exception(f"No function to convert {data_type}")
