  






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