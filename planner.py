"""
使用数学公式进行建模，对球拍指令进行规划。使用PyTorch，支持并行计算。
需要实现以下三个功能：
1. reset时根据球的初始位置和速度，计算与球桌的碰撞时间、接近击球平面时的位置和速度
2. 与球桌碰撞完毕后，接近击球平面时的位置和速度
3. 为了将到达击球平面的球给打回去，计算球拍的位置和速度，作为网络的输入指令（球拍的朝向正反手都可以，让他自己学）
"""

import torch


def compute_land(ball_pos, ball_vel, landing_h=0.795):
    # 与左右发球无关，只与球桌高度有关
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
    # 计算paddle vel的时候，需要判断target pos是否在范围内，可以直接根据发球采样的方式
    # hit_pos都是在右边，不用考虑side choice
    # TODO: 必须采样到到达发球范围的速度,简单的方式是直接没在范围内就重新采样
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
    # TODO： v_x 还有一个下界，必须要过网
    A = net_h - hit_pos[:, 2]
    B = -v_z * (net_x - hit_pos[:, 0])
    C = 0.5 * gravity * (net_x - hit_pos[:, 0])**2
    v_x_min = find_low_v_x(A, B, C)

    v_lower[:,0] = torch.minimum(v_lower[:,0], v_x_min)

    # TODO：判断此球速是否到达对面的hit plane，不然就一直重新采样
    v_out = torch.rand((n_envs, 3)).to(device) * (v_upper - v_lower) + v_lower
    valid_mask = torch.zeros(n_envs, device=device, dtype=torch.bool)
    
    # 循环采样直到所有样本的 hit_plane_pos 第0维都大于1.1
    max_iterations = 50  # 防止无限循环
    iteration = 0
    
    while not valid_mask.all() and iteration < max_iterations:
        # 判断此球速是否到达对面的hit plane
        t_land, land_pos, land_vel = compute_land(hit_pos, v_out)
        bounce_vel = land_vel.clone()
        bounce_vel[:, 2] = -bounce_vel[:, 2]
        hit_plane_pos, _, _ = compute_hit_pos(t_land, land_pos, bounce_vel, hit_plane_x=-1.8)
        
        # 检查哪些样本满足条件
        valid_mask = (hit_plane_pos[:, 2] > 1.1) & (hit_plane_pos[:, 2] < 1.35) & (hit_plane_pos[:, 1] > -0.7) & (hit_plane_pos[:, 1] < 0.7)
        
        # 对不满足条件的样本重新采样
        if not valid_mask.all():
            invalid_indices = ~valid_mask
            n_invalid = invalid_indices.sum().item()
            if n_invalid > 0:
                # 只对不满足条件的样本重新采样, 从v_z开始采样
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
        
        # print(f"第{iteration}次采样")
        iteration += 1
    



    # # 暴力搜索的方式解决碰网问题
    # g = torch.tensor([0, 0, -9.81]).to(hit_pos.device)
    # target_pos = torch.tensor([-0.85, 0.04, 0.795]).to(hit_pos.device)

    # t_range = torch.linspace(t_min, t_max, n_steps, device=hit_pos.device)

    # # 计算期望的出射速度：根据目标位置和飞行时间反推
    # # v_out = (target_pos - hit_pos) / t_flight - 0.5 * g * t_flight
    # # 扩展维度以支持广播: hit_pos (1024, 3), target_pos (3,) -> (1024, 3)
    # # 然后扩展为 (1024, 1, 3) 与 t_range (1, 20, 1) 广播得到 (1024, 20, 3)
    # pos_diff = (target_pos - hit_pos).unsqueeze(1)  # (1024, 1, 3)
    # t_range_expanded = t_range.view(1, -1, 1)  # (1, 20, 1)
    # g_expanded = g.view(1, 1, -1)  # (1, 1, 3)
    # v_out = pos_diff / t_range_expanded - 0.5 * g_expanded * t_range_expanded  # (1024, 20, 3)

    # t_net = (net_x - hit_pos[:, 0]).view(-1,1) / v_out[:, :, 0]

    # z_at_net = hit_pos[:,2].view(-1,1) + v_out[:, :, 2] * t_net + 0.5 * g[2] * t_net**2
    # cross_net_flag = z_at_net > net_h

    # # 选择每个环境中第一个True的最小索引，输出形状为(1024,)
    # # 使用argmax找到每行第一个True的索引（True=1, False=0，argmax返回第一个最大值的索引）
    # first_true_idx = cross_net_flag.int().argmax(dim=1)  # (1024,)
    # v_out = v_out[torch.arange(v_out.shape[0], device=v_out.device), first_true_idx]  # (1024, 3)
    

    # 计算碰撞法向量（单位向量）：指向速度变化的方向
    # u = (v_out - v_in) / ||v_out - v_in||
    u = (v_out - v_in) / torch.norm(v_out - v_in, dim=-1, keepdim=True)
    
    # 计算球拍速度：
    # v_racket = (v_out · u + v_in · u) / (1 + C_r) * u
    # 其中 v_out · u 和 v_in · u 分别是出射速度和入射速度在法向量方向上的投影
    v_racket = (torch.sum(v_out * u, dim=-1, keepdim=True) + torch.sum(v_in * u, dim=-1, keepdim=True)) / (1 + C_r) * u

    # TODO: 添加一点随机化以克服gap

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
    计算将 from_vec 旋转到 to_vec 的四元数。
    这段代码的思路是计算绝对旋转，即从被旋转物体的初始未旋转位置(局部和全局坐标系重合)开始计算，计算出一个绝对旋转四元数
    
    Args:
        from_vec: (n, 3) tensor
        to_vec: (n, 3) tensor
    
    Returns:
        quat: (n, 4) tensor, 格式为 (w, x, y, z)
    """
    # 归一化输入向量
    from_vec = from_vec / torch.norm(from_vec, dim=-1, keepdim=True)
    to_vec = to_vec / torch.norm(to_vec, dim=-1, keepdim=True)

    # 计算旋转轴和角度
    axis = torch.cross(from_vec, to_vec, dim=-1)

    angle = torch.acos(torch.clamp(torch.sum(from_vec * to_vec, dim=-1), -1.0, 1.0))

    # 计算轴的长度
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    
    # 处理向量平行或反向的特殊情况
    # 几乎平行的情况 (axis_norm < 1e-6 且 angle < 0.1)
    parallel_mask = (axis_norm.squeeze(-1) < 1e-6) & (angle < 0.1)
    
    # 反向的情况 (axis_norm < 1e-6 且 angle >= 0.1)
    opposite_mask = (axis_norm.squeeze(-1) < 1e-6) & (angle >= 0.1)
    
    # 正常情况
    normal_mask = ~(parallel_mask | opposite_mask)
    
    # 归一化轴（只在正常情况下）
    axis_normalized = axis / (axis_norm + 1e-8)  # 添加小值避免除零
    
    # 计算四元数 (w, x, y, z)
    half_angle = angle / 2
    quat = torch.zeros(from_vec.shape[0], 4, device=from_vec.device, dtype=from_vec.dtype)
    
    # 正常情况：quat = [cos(angle/2), axis * sin(angle/2)]
    quat[normal_mask, 0] = torch.cos(half_angle[normal_mask])
    quat[normal_mask, 1:4] = axis_normalized[normal_mask] * torch.sin(half_angle[normal_mask]).unsqueeze(-1)
    
    # 几乎平行：无旋转 [1, 0, 0, 0]
    quat[parallel_mask, 0] = 1.0
    
    # 反向：绕X轴旋转180度 [0, 1, 0, 0]
    quat[opposite_mask, 1] = 1.0
    
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
    
    # 如果球的y坐标在球桌外边，说明出界，击球失败；
    # 如果球的z坐标低于网的高度，说明撞网，击球失败
    fail = torch.zeros_like(ball_pos[:, 1], dtype=torch.bool)
    fail |= (land_pos[:, 1] > 0.80)
    fail |= (land_pos[:, 1] < -0.72)
    fail |= (land_pos[:, 2] < net_h)

    return fail