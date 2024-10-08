import roma
import torch


def rotMat2pos(rot_mat, start_pos, xpos_start, identity_focus, fix_determinant=True):
    """
    Transforms a batch of vectors by a rotation matrix and translation vector.

    Input:
        rot_mat: Original predictions (euclidean motion)
            - Shape for non-recurrent network: (batch, 12)
            - Shape for recurrent network: (batch, frames, 12)
        start_pos: Start position of simulation
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, 24)
        xpos_start: Start position of centroid
            - If necessary: (batch_size, 3)
            - If not necessary: None

    Output:
        Converted eucledian motion to current xyz position
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, frames, 24)
    """
    device = rot_mat.device
    if len(rot_mat.shape) == 2:
        if xpos_start is None:
            xpos_start = 0
        else:
            xpos_start = xpos_start.reshape(-1, 1, 3)

        rotations = rot_mat[:, :9].reshape(-1, 3, 3)  # [Batch_size, 3, 3]
        # Ensure prediction represents rotation matrix
        if identity_focus:
            identity = (
                torch.eye(3)[None, ...].repeat(rotations.shape[0], 1, 1).to(device)
            )
            rotations += identity

        u, _, vT = torch.linalg.svd(rotations)
        true_rotations = torch.bmm(u, vT)

        start_origin = (
            start_pos.reshape(-1, 8, 3) - xpos_start
        ).mT  # [batch_size, 3, 8]

        mult = torch.bmm(true_rotations, start_origin).mT  # [batch_size, 8, 3]
        out = mult + rot_mat[:, 9:].reshape(-1, 1, 3) + xpos_start  # [batch_size, 8, 3]

        return out.flatten(start_dim=1)  # [batch_size, 24]
    # In case of LSTM/GRU
    else:
        if xpos_start is None:
            xpos_start = 0
        else:
            xpos_start = xpos_start.flatten(end_dim=1)[:, None, :]

        rotations = rot_mat[..., :9].reshape(
            rot_mat.shape[0], rot_mat.shape[1], 3, 3
        )  # [Batch_size, frames, 3, 3]
        flat_rotations = rotations.flatten(end_dim=1)  # [Batch_size x frames, 3, 3]

        if identity_focus:
            identity = (
                torch.eye(3)[None, ...].repeat(flat_rotations.shape[0], 1, 1).to(device)
            )
            # print(identity.shape)
            flat_rotations += identity

        # Ensure prediction represents rotation matrix
        u, _, vT = torch.linalg.svd(flat_rotations)
        true_rotations = torch.bmm(u, vT)
        if fix_determinant:
            det = torch.linalg.det(true_rotations)
            # print(det)
            mask = torch.isclose(det, torch.tensor(-1.0), atol=0.01)
            if torch.sum(mask) > 0:
                u_new = u.clone()
                u_new[mask, :, -1] = -u[mask, :, -1]
                true_rotations = torch.bmm(u_new, vT)
                # det_2 = torch.linalg.det(true_rotations)
                # mask_2 = torch.isclose(det_2, torch.tensor(-1.0), atol=0.01)
                # assert (
                #     torch.sum(mask_2) == 0
                # ), f"TRIED AGAIN.. BUT FAILED {sum(mask_2).item()}/{len(mask_2)}"

        start_origin = (
            start_pos.reshape(-1, rot_mat.shape[1], 8, 3).flatten(end_dim=1)
            - xpos_start
        ).mT  # [Batch_size x frames, 3, 8]

        mult = torch.bmm(true_rotations, start_origin).mT  # [Batch_size x frames, 8, 3]
        out = (
            mult + xpos_start + rot_mat[..., 9:].flatten(end_dim=1)[:, None, :]
        ).flatten(
            start_dim=1
        )  # [Batch_size x frames, 24]
        out = out.reshape(rot_mat.shape[0], rot_mat.shape[1], -1)
        return out  # [Batch_size, frames, 24]


def fast_rotVecQuat(v, q):
    """
    Returns rotated vectors v by quaternions q.
    Input:
        v: vector to be rotated
            shape: (* x 24)
        q: quaternion to rotate v
            shape: (* x 4)

    Output:
        Rotated batch of vectors v by a batch of quaternions q.
    """

    device = v.device

    # Ensure prediction represents pure rotation
    q_norm = torch.div(q.T, torch.norm(q, dim=-1)).T

    # Swap columns for roma calculations (bi, cj, dk, a)
    q_new1 = torch.index_select(q_norm, 1, torch.tensor([1, 2, 3, 0], device=device))

    rotated_v = torch.bmm(
        roma.unitquat_to_rotmat(q_new1), v.reshape(-1, 8, 3).mT
    ).mT.to(device)

    return rotated_v


def quat2pos(quat, start_pos, xpos_start, identity_focus, add_identity=True):
    """
    Input:
        - quat: Original predictions (quaternion motion)
            - Shape for non-recurrent network: (batch, 7)
            - Shape for recurrent network: (batch, frames, 7)
        - start_pos: Start position of simulation
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, 24)
        - xpos_start: Start position of centroid
            - If necessary: (batch, 3)
            - If not necessary: None

    Output:
        - Converted quaternion to current xyz position
            (batch, 24) or (batch, frames, 24)
    """
    device = quat.device
    if len(quat.shape) == 2:
        if xpos_start is None:
            xpos_start = 0
        else:
            xpos_start = xpos_start.reshape(-1, 1, 3)

        if identity_focus and add_identity:
            identity = (
                torch.FloatTensor([1, 0, 0, 0, 0, 0, 0])[None, :].repeat(
                    quat.shape[0], 1
                )
            ).to(device)
            quat += identity

        start_pos_shape = start_pos.shape
        start_origin = (start_pos.reshape(-1, 8, 3) - xpos_start).reshape(
            start_pos_shape
        )
        rotated_start = fast_rotVecQuat(
            start_origin,
            quat[:, :4],
        )
        repeated_trans = quat[:, 4:][:, None, :]
        out = rotated_start + repeated_trans + xpos_start
        return out.flatten(start_dim=1)
    else:
        if xpos_start is None:
            xpos_start = 0
        else:
            # For visualisation
            if len(xpos_start.shape) != 3:
                xpos_start = xpos_start
            xpos_start = xpos_start.flatten(end_dim=1)[:, None, :]
        quat_flat = quat.flatten(end_dim=1)

        if identity_focus and add_identity:
            identity = (
                torch.FloatTensor([1, 0, 0, 0, 0, 0, 0])[None, :]
                .repeat(quat_flat.shape[0], 1)
                .to(device)
            )
            quat_flat += identity

        # For visualisation
        repeated_start_pos = start_pos.flatten(end_dim=1).reshape(-1, 8, 3)
        start_origin = (repeated_start_pos - xpos_start).flatten(start_dim=1)

        # Rotate start by quaternion
        rotated_start = fast_rotVecQuat(start_origin, quat_flat[:, :4])
        # Add Translation
        out = (rotated_start + xpos_start + quat_flat[:, 4:][:, None, :]).flatten(
            start_dim=1
        )
        # Fix shape
        out = out.reshape(quat.shape[0], quat.shape[1], out.shape[-1])
        return out


def log_quat2pos(log_quat, start_pos, start_xpos, identity_focus):
    """
    Input:
        - log_quat: Original predictions (log quaternion motion)
            - Shape for non-recurrent network: (batch, 7)
            - Shape for recurrent network: (batch, frames, 7)
        - start_pos: Start position of simulation
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, 24)
        - xpos_start: Start position of centroid
            - If necessary: (batch, 3)
            - If not necessary: None

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

        return quat2pos(
            full_quat, start_pos, start_xpos, identity_focus, add_identity=False
        )

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

        return quat2pos(
            full_quat, start_pos, start_xpos, identity_focus, add_identity=False
        )


def dualQ_normalize(dualQ):
    A = 1 / torch.sqrt(
        dualQ[..., 0] ** 2
        + dualQ[..., 1] ** 2
        + dualQ[..., 2] ** 2
        + dualQ[..., 3] ** 2
    )
    B = (
        (
            dualQ[..., 4] * dualQ[..., 0]
            - (
                -dualQ[..., 5] * dualQ[..., 1]
                - dualQ[..., 6] * dualQ[..., 2]
                - dualQ[..., 7] * dualQ[..., 3]
            )
        )
        * A
        * A
        * A
    )
    res = torch.zeros_like(dualQ)
    res[..., 0] = A * dualQ[..., 0]
    res[..., 1] = A * dualQ[..., 1]
    res[..., 2] = A * dualQ[..., 2]
    res[..., 3] = A * dualQ[..., 3]
    res[..., 4] = A * dualQ[..., 4] - B * dualQ[..., 0]
    res[..., 5] = -(A * -dualQ[..., 5] + B * dualQ[..., 1])
    res[..., 6] = -(A * -dualQ[..., 6] + B * dualQ[..., 2])
    res[..., 7] = -(A * -dualQ[..., 7] + B * dualQ[..., 3])
    return res


def dualQ2pos(dualQ, start_pos, start_xpos, identity_focus, add_identity=True):
    """
    Input:
        - dualQ: Original predictions (Dual quaternion)
            - Shape for non-recurrent network: (batch, 8)
            - Shape for recurrent network: (batch, frames, 8)
        - start_pos: Start position of simulation
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, 24)
        - xpos_start: Start position of centroid
            - If necessary: (batch, 3)
            - If not necessary: None

    Output:
        - Converted Dual-quaternion to current position
            (batch x 24) or
            (batch x frames x 24)
    """
    device = dualQ.device

    # Ensure prediction represents pure rotation
    if identity_focus and add_identity:
        identity = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).to(device)
        if len(dualQ.shape) == 3:
            identity = identity[None, None, :].repeat(dualQ.shape[0], 1, 1)
        else:
            identity = identity[None, :].repeat(dualQ.shape[0], 1)
        dualQ += identity

    dualQ = dualQ_normalize(dualQ)

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

    converted_pos = quat2pos(
        quaternion, start_pos, start_xpos, identity_focus, add_identity=False
    )
    return converted_pos


def log_dualQ2pos(logDualQ_in, start_pos, start_xpos, identity_focus):
    """
    Code from Normalization, Square Roots, and the Exponential and Logarithmic Maps
                in Geometric Algebras of Less than 6D (arXiv:2206.07496v2)

    Input bivector (6 numbers) returns position by first calculating the dual quaternion = exp(log_dualQ).
    (17 mul, 8 add, 2 div, 1 sincos, 1 sqrt)

    Input:
        - log_dualQ: Original predictions (logarithm of dual quaternion)
            - Shape for non-recurrent network: (batch, 6)
            - Shape for recurrent network: (batch, frames, 6)
        - start_pos: Start position of simulation
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, 24)
        - xpos_start: Start position of centroid
            - If necessary: (batch, 3)
            - If not necessary: None

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
    mask = l == 0
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
    dualQ[mask, :] = alternative[mask, :]

    return dualQ2pos(
        dualQ.reshape(out_shape),
        start_pos,
        start_xpos,
        identity_focus,
        add_identity=False,
    )


def diff_pos_start2pos(true_preds, start_pos):
    """
    Input:
        - true_preds: Original predictions (difference compared to start)
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, frames, 24)
        - start_pos: Start position of simulation
            - Shape for non-recurrent network: (batch, 24)
            - Shape for recurrent network: (batch, 24)
    Output:
        - Converted difference to current position
    """

    if len(true_preds.shape) == 2:
        true_preds = true_preds[:, None, :]
    if len(start_pos.shape) == 2:
        start_pos = start_pos[:, None, :]

    result = start_pos + true_preds
    return result.squeeze()


def convert(true_preds, start_pos, data_type, xpos_start, identity=False):
    """
    Converts true predictions given data type.
    Input:
        - true_preds: Prediction of model
        - start_pos: Start position of simulation
        - data_type: Datatype of predictions
    Output:
        - Converted true predictions.
    """
    if data_type[-3:] == "ori":
        xpos_start = None
    if data_type == "pos" or data_type == "pos_norm":
        return true_preds
    elif data_type[:7] == "rot_mat":
        return rotMat2pos(true_preds, start_pos, xpos_start, identity)
    elif data_type[:4] == "quat":
        return quat2pos(true_preds, start_pos, xpos_start, identity)
    elif data_type[:8] == "log_quat":
        return log_quat2pos(true_preds, start_pos, xpos_start, identity)
    elif data_type[:9] == "dual_quat":
        return dualQ2pos(true_preds, start_pos, xpos_start, identity)
    elif data_type[:9] == "log_dualQ":
        return log_dualQ2pos(true_preds, start_pos, xpos_start, identity)
    elif data_type[:8] == "pos_diff":
        return diff_pos_start2pos(true_preds, start_pos)
    raise Exception(f"No function to convert {data_type}")
