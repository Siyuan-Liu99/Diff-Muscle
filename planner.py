"""
Physics-based paddle planner using PyTorch for parallel computation.

Provides three core functions:
1. On reset: compute ball-table collision time and ball state when it reaches the hit plane.
2. Post-bounce: compute ball position and velocity as it approaches the hit plane.
3. Compute target paddle position and velocity needed to return the incoming ball
   (forehand/backhand is left for the policy to learn).
"""

import torch


def compute_land(ball_pos, ball_vel, landing_h=0.795):
    g = -9.81
    t_land = (-ball_vel[:, 2] - (ball_vel[:, 2] ** 2 - 2 * g * (ball_pos[:, 2] - landing_h)) ** 0.5) / g

    land_pos = torch.zeros_like(ball_pos)
    land_pos[:, 0:2] = ball_pos[:, 0:2] + ball_vel[:, 0:2] * t_land.unsqueeze(-1)
    land_pos[:, 2] = ball_pos[:, 2] + ball_vel[:, 2] * t_land + 0.5 * g * t_land**2
    land_vel = torch.zeros_like(ball_vel)
    land_vel[:, 0:2] = ball_vel[:, 0:2]
    land_vel[:, 2] = ball_vel[:, 2] + g * t_land
    return t_land, land_pos, land_vel


def compute_hit_pos(t_land, bounce_pos, bounce_vel, hit_plane_x=1.8):
    g = -9.81
    t_hit = (hit_plane_x - bounce_pos[:, 0]) / bounce_vel[:, 0]
    t_total_round = torch.round(t_land + t_hit, decimals=2)
    t_hit_round = t_total_round - t_land

    hit_pos = torch.zeros_like(bounce_pos)
    hit_pos[:, 0:2] = bounce_pos[:, 0:2] + bounce_vel[:, 0:2] * t_hit_round.unsqueeze(-1)
    hit_pos[:, 2] = 0.5 * g * t_hit_round**2 + bounce_vel[:, 2] * t_hit_round + bounce_pos[:, 2]

    hit_vel = torch.zeros_like(bounce_vel)
    hit_vel[:, 0:2] = bounce_vel[:, 0:2]
    hit_vel[:, 2] = bounce_vel[:, 2] + g * t_hit_round

    return hit_pos, hit_vel, t_total_round


def compute_paddle_vel(hit_pos, v_in, table_area_upper, table_area_lower, device, C_r=1.0, net_h=0.95+0.05, net_x=0.0):
    # TODO: resample until v_out lands within the target serving area
    n_envs = hit_pos.shape[0]
    gravity = 9.81

    # sample V_z
    v_z = torch.rand((n_envs,)).to(device) * 0.2 + 0.6

    # compute t
    a = -0.5 * gravity
    b = v_z
    c = hit_pos[:, 2] - table_area_upper[2]

    discriminant = b**2 - 4 * a * c
    t = (-b - discriminant**0.5) / (2 * a)

    # compute V_x and V_y
    v_upper = torch.stack(
            [(table_area_upper[0] - hit_pos[:, 0]) / t, (table_area_upper[1] - hit_pos[:, 1]) / t, v_z], dim=-1
        )
    v_lower = torch.stack(
            [(table_area_lower[0] - hit_pos[:, 0]) / t, (table_area_lower[1] - hit_pos[:, 1]) / t, v_z], dim=-1
        )
    # TODO: v_x has a lower bound - the ball must clear the net
    A = net_h - hit_pos[:, 2]
    B = -v_z * (net_x - hit_pos[:, 0])
    C = 0.5 * gravity * (net_x - hit_pos[:, 0])**2
    v_x_min = find_low_v_x(A, B, C)

    v_lower[:,0] = torch.minimum(v_lower[:,0], v_x_min)

    # Resample until all trajectories reach the opponent's hit plane
    v_out = torch.rand((n_envs, 3)).to(device) * (v_upper - v_lower) + v_lower
    valid_mask = torch.zeros(n_envs, device=device, dtype=torch.bool)

    max_iterations = 50
    iteration = 0

    while not valid_mask.all() and iteration < max_iterations:
        t_land, land_pos, land_vel = compute_land(hit_pos, v_out)
        bounce_vel = land_vel.clone()
        bounce_vel[:, 2] = -bounce_vel[:, 2]
        hit_plane_pos, _, _ = compute_hit_pos(t_land, land_pos, bounce_vel, hit_plane_x=-1.8)

        valid_mask = (hit_plane_pos[:, 2] > 1.1) & (hit_plane_pos[:, 2] < 1.35) & (hit_plane_pos[:, 1] > -0.7) & (hit_plane_pos[:, 1] < 0.7)

        if not valid_mask.all():
            invalid_indices = ~valid_mask
            n_invalid = invalid_indices.sum().item()
            if n_invalid > 0:
                v_z_new = torch.rand((n_invalid,)).to(device) * 0.2 + 0.6
                b = v_z_new
                c = hit_pos[invalid_indices, 2] - table_area_upper[2]

                discriminant = b**2 - 4 * a * c
                t = (-b - discriminant**0.5) / (2 * a)

                v_upper = torch.stack(
                        [(table_area_upper[0] - hit_pos[invalid_indices, 0]) / t, (table_area_upper[1] - hit_pos[invalid_indices, 1]) / t, v_z_new], dim=-1
                    )
                v_lower = torch.stack(
                        [(table_area_lower[0] - hit_pos[invalid_indices, 0]) / t, (table_area_lower[1] - hit_pos[invalid_indices, 1]) / t, v_z_new], dim=-1
                    )
                # v_lower.shape[0] = len(invalid_indices)

                A = net_h - hit_pos[invalid_indices, 2]
                B = -v_z_new * (net_x - hit_pos[invalid_indices, 0])
                C = 0.5 * gravity * (net_x - hit_pos[invalid_indices, 0])**2
                v_x_min = find_low_v_x(A, B, C)

                v_lower[:, 0] = torch.minimum(v_lower[:, 0], v_x_min)

                v_out_new = torch.rand((n_invalid, 3)).to(device) * (v_upper - v_lower) + v_lower
                v_out[invalid_indices] = v_out_new

        iteration += 1

    # Collision normal (unit vector pointing in the direction of velocity change):
    # u = (v_out - v_in) / ||v_out - v_in||
    u = (v_out - v_in) / torch.norm(v_out - v_in, dim=-1, keepdim=True)

    # Paddle velocity from elastic collision model:
    # v_racket = (dot(v_out, u) + dot(v_in, u)) / (1 + C_r) * u
    v_racket = (torch.sum(v_out * u, dim=-1, keepdim=True) + torch.sum(v_in * u, dim=-1, keepdim=True)) / (1 + C_r) * u

    # TODO: add small randomization to improve robustness

    return v_racket


def find_low_v_x(A, B, C):
    delta = B**2 - 4 * A * C

    if (delta < 0).any():
        mask_delta_neg = delta < 0
        root1 = (-B + delta**0.5) / (2 * A)
        root2 = torch.zeros_like(root1)
        return torch.where(mask_delta_neg, root2, root1)

    root1 = (-B + delta**0.5) / (2 * A)
    return root1
    



def vec_to_quat(from_vec, to_vec):
    """
    Compute the quaternion that rotates from_vec to to_vec.

    Computes an absolute rotation quaternion, starting from the object's
    unrotated pose (local and global frames coincide).

    Args:
        from_vec: (n, 3) tensor
        to_vec: (n, 3) tensor

    Returns:
        quat: (n, 4) tensor in (w, x, y, z) format
    """
    from_vec = from_vec / torch.norm(from_vec, dim=-1, keepdim=True)
    to_vec = to_vec / torch.norm(to_vec, dim=-1, keepdim=True)

    axis = torch.cross(from_vec, to_vec, dim=-1)
    angle = torch.acos(torch.clamp(torch.sum(from_vec * to_vec, dim=-1), -1.0, 1.0))

    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Nearly parallel (axis_norm < 1e-6 and angle < 0.1): no rotation
    parallel_mask = (axis_norm.squeeze(-1) < 1e-6) & (angle < 0.1)
    # Anti-parallel (axis_norm < 1e-6 and angle >= 0.1): 180° rotation about X
    opposite_mask = (axis_norm.squeeze(-1) < 1e-6) & (angle >= 0.1)
    normal_mask = ~(parallel_mask | opposite_mask)

    axis_normalized = axis / (axis_norm + 1e-8)

    half_angle = angle / 2
    quat = torch.zeros(from_vec.shape[0], 4, device=from_vec.device, dtype=from_vec.dtype)

    quat[normal_mask, 0] = torch.cos(half_angle[normal_mask])
    quat[normal_mask, 1:4] = axis_normalized[normal_mask] * torch.sin(half_angle[normal_mask]).unsqueeze(-1)
    quat[parallel_mask, 0] = 1.0   # identity
    quat[opposite_mask, 1] = 1.0   # 180° about X-axis

    return quat


def compute_paddle_pos(paddle_ori, hit_pos):
    # get paddle pos
    # paddle_ori is (n, 4) tensor with format [w, x, y, z]
    # Convert quaternion to rotation matrix
    w, x, y, z = paddle_ori[:, 0], paddle_ori[:, 1], paddle_ori[:, 2], paddle_ori[:, 3]
    
    # Quaternion to rotation matrix conversion
    rot_mat = torch.zeros(paddle_ori.shape[0], 3, 3, device=paddle_ori.device, dtype=paddle_ori.dtype)
    rot_mat[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_mat[:, 0, 1] = 2 * (x*y - w*z)
    rot_mat[:, 0, 2] = 2 * (x*z + w*y)
    rot_mat[:, 1, 0] = 2 * (x*y + w*z)
    rot_mat[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_mat[:, 1, 2] = 2 * (y*z - w*x)
    rot_mat[:, 2, 0] = 2 * (x*z - w*y)
    rot_mat[:, 2, 1] = 2 * (y*z + w*x)
    rot_mat[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    # Local paddle offset
    pad_local = torch.tensor([-0.07, 0, 0], device=paddle_ori.device, dtype=paddle_ori.dtype)
    # Batch matrix multiplication: (n, 3, 3) @ (3,) -> (n, 3)
    pad_world = torch.matmul(rot_mat, pad_local)
    p_paddle = hit_pos - pad_world    

    return p_paddle


def compute_land_net(ball_pos, ball_vel, net_x=0.0, net_h=0.95):
    g = -9.81
    t_land_net = (net_x - ball_pos[:, 0]) / ball_vel[:, 0]
    land_pos = torch.zeros_like(ball_pos)
    land_pos[:, 0:2] = ball_pos[:, 0:2] + ball_vel[:, 0:2] * t_land_net.unsqueeze(-1)
    land_pos[:, 2] = ball_pos[:, 2] + ball_vel[:, 2] * t_land_net + 0.5 * g * t_land_net**2
    
    # Ball is out of bounds (y outside table width) or hits the net (z below net height)
    fail = torch.zeros_like(ball_pos[:, 1], dtype=torch.bool)
    fail |= (land_pos[:, 1] > 0.80)
    fail |= (land_pos[:, 1] < -0.72)
    fail |= (land_pos[:, 2] < net_h)

    return fail