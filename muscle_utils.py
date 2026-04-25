from typing import Callable
import warp as wp
import mujoco
import mujoco_warp as mjw
from mujoco_warp._src.util_misc import muscle_bias, muscle_gain
from mujoco_warp._src.types import vec10, MJ_MINVAL
import torch


@wp.kernel
def muscle_bias_kernel(
    lengths: wp.array(dtype=float),
    lengthranges: wp.array(dtype=wp.vec2),
    acc0s: wp.array(dtype=float),
    prms: wp.array(dtype=vec10),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = muscle_bias(lengths[tid], lengthranges[tid], acc0s[tid], prms[tid])


@wp.kernel
def muscle_gain_kernel(
    lengths: wp.array(dtype=float),
    velocities: wp.array(dtype=float),
    lengthranges: wp.array(dtype=wp.vec2),
    acc0s: wp.array(dtype=float),
    prms: wp.array(dtype=vec10),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = muscle_gain(lengths[tid], velocities[tid], lengthranges[tid], acc0s[tid], prms[tid])


def get_target_actuator_length(
    model: mjw.Model,
    fk_data: mjw.Data,
    qpos: torch.Tensor,
    equalities: list[tuple[int, int, list[float]]],
    action_to_qpos: torch.Tensor,
    fk_forward: Callable,
):
    fk_data.qpos[:, action_to_qpos] = qpos

    for id1, id2, eq_data in equalities:
        qpos_id1 = model.jnt_qposadr[id1]
        qpos_id2 = model.jnt_qposadr[id2]

        if id2 == -1:
            fk_data.qpos[:, qpos_id1] = model.qpos0[:, qpos_id1] + eq_data[0]
        else:
            fk_data.qpos[:, qpos_id1] = torch.clamp(
                model.qpos0[:, qpos_id1]
                + eq_data[0]
                + eq_data[1] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2])
                + eq_data[2] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2]) ** 2
                + eq_data[3] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2]) ** 3
                + eq_data[4] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2]) ** 4,
                model.jnt_range[:, id1, 0],
                model.jnt_range[:, id1, 1],
            )

    fk_forward()
    return fk_data.actuator_length


def target_length_to_activations(
    model: mjw.Model,
    data: mjw.Data,
    target_length: torch.Tensor,
    kp_scale: torch.Tensor | float,
    kd_scale: torch.Tensor | float,
):

    # 获取设备的device, 在多gpu训练的时候保证device和env对齐
    device = target_length.device
    wp_device = wp.get_device(str(device))
    
    muscle_indices = model.actuator_dyntype == 4  # mujoco.mjtDyn.mjDYN_MUSCLE

    length = data.actuator_length[:, muscle_indices]
    lengthrange = model.actuator_lengthrange[muscle_indices]  # w/o batch dim
    velocity = data.actuator_velocity[:, muscle_indices]
    peak_force = model.actuator_biasprm[:, muscle_indices, 2]  # w/ batch dim

    force = (
        (kp_scale * (target_length[:, muscle_indices] - length) - kd_scale * kp_scale * velocity)
        * peak_force
        / (lengthrange[:, 1] - lengthrange[:, 0])
    )
    clipped_force = torch.clamp(force, -peak_force, torch.zeros_like(peak_force))

    prmb = model.actuator_biasprm[:, muscle_indices, :10]  # w/ batch dim
    prmg = model.actuator_gainprm[:, muscle_indices, :10]  # w/ batch dim
    acc0 = model.actuator_acc0[muscle_indices].unsqueeze(0).repeat(prmb.shape[0], 1)  # add batch dim
    lengthrange = lengthrange.unsqueeze(0).repeat(prmb.shape[0], 1, 1)  # add batch dim

    with wp.ScopedDevice(wp_device):
        bias = wp.zeros(prmb.shape[0] * prmb.shape[1], dtype=float, device=wp_device)
        wp.launch(
            muscle_bias_kernel,
            dim=prmb.shape[0] * prmb.shape[1],
            inputs=[length.reshape(-1), lengthrange.reshape(-1, 2), acc0.reshape(-1), prmb.reshape(-1, 10), bias],
            device=bias.device,
        )

        gain = wp.zeros(prmb.shape[0] * prmb.shape[1], dtype=float, device=wp_device)
        wp.launch(
            muscle_gain_kernel,
            dim=prmb.shape[0] * prmb.shape[1],
            inputs=[
                length.reshape(-1),
                velocity.reshape(-1),
                lengthrange.reshape(-1, 2),
                acc0.reshape(-1),
                prmg.reshape(-1, 10),
                gain,
            ],
            device=gain.device,
        )

        bias = wp.to_torch(bias).reshape(prmb.shape[0], prmb.shape[1])
        gain = wp.to_torch(gain).reshape(prmb.shape[0], prmb.shape[1])
        gain = torch.clamp(gain, max=-1)
        activations = torch.clamp((clipped_force - bias) / gain, 0, 1)
    return activations, bias, gain, clipped_force


def calculate_vae_muscle_act(
    model: mjw.Model,
    data: mjw.Data,
    target_length: torch.Tensor,
    kp: float = 10.0,
    kd: float = 1.0,
) -> torch.Tensor:
    """
    计算 MuscleVAE 所需的肌肉激活值。
    
    通过 PD 控制器计算理想收缩力，并利用 MuJoCo 的 FLV 曲线参数进行逆动力学解算，
    将目标肌肉长度转换为肌肉激活信号。
    
    Args:
        model: MuJoCo Warp 模型，包含肌肉参数
        data: MuJoCo Warp 数据，包含当前肌肉状态
        target_length: 目标肌肉长度 Tensor [batch_size, num_muscles]
                       由策略输出映射得到
        kp: PD 控制器的比例增益 (默认: 10.0)
        kd: PD 控制器的微分增益 (默认: 1.0)
    
    Returns:
        activations: 肌肉激活值 Tensor [batch_size, num_muscles]，范围 [0, 1]
    
    Notes:
        计算流程:
        1. PD 力计算: f_pd = kp * (l_target - l_current) - kd * v_current
        2. 力归一化: f_normalized = f_pd * F0 / (l_max - l_min)
        3. 物理裁剪: f_clipped = clamp(f_normalized, -F0, 0)  # 肌肉只能拉不能推
        4. 逆动力学: activation = (f_clipped - bias) / gain
           其中 bias 和 gain 由 MuJoCo FLV 曲线参数计算得到
    """
    # 获取设备信息，确保多 GPU 训练时 device 对齐
    device = target_length.device
    wp_device = wp.get_device(str(device))
    
    # 获取肌肉执行器索引 (dyntype == 4 表示 mjDYN_MUSCLE)
    muscle_indices = model.actuator_dyntype == 4
    
    # ========== Step 1: 提取当前肌肉状态 ==========
    # 当前肌肉长度 [batch_size, num_muscles]
    length = data.actuator_length[:, muscle_indices]
    # 当前肌肉收缩速度 [batch_size, num_muscles]
    velocity = data.actuator_velocity[:, muscle_indices]
    # 肌肉长度范围 [num_muscles, 2] (无 batch 维度)
    lengthrange = model.actuator_lengthrange[muscle_indices]
    # 最大等长力 F0 [batch_size, num_muscles]
    F0 = model.actuator_biasprm[:, muscle_indices, 2]
    
    # ========== Step 2: PD 力计算 ==========
    # f_pd = kp * (l_target - l_current) - kd * v_current
    # 归一化：乘以 F0 / (l_max - l_min) 将位置误差转换为力
    length_error = target_length[:, muscle_indices] - length
    f_pd = (kp * length_error - kd * velocity) * F0 / (lengthrange[:, 1] - lengthrange[:, 0])
    
    # ========== Step 3: 物理裁剪 ==========
    # 肌肉只能产生拉力（负值），不能推，且不能超过最大等长力
    f_clipped = torch.clamp(f_pd, min=-F0, max=torch.zeros_like(F0))
    
    # ========== Step 4: 逆动力学解算 ==========
    # 获取 FLV 曲线参数
    prmb = model.actuator_biasprm[:, muscle_indices, :10]  # bias 参数 [batch, num_muscles, 10]
    prmg = model.actuator_gainprm[:, muscle_indices, :10]  # gain 参数 [batch, num_muscles, 10]
    acc0 = model.actuator_acc0[muscle_indices].unsqueeze(0).repeat(prmb.shape[0], 1)  # [batch, num_muscles]
    lengthrange_batched = lengthrange.unsqueeze(0).repeat(prmb.shape[0], 1, 1)  # [batch, num_muscles, 2]
    
    batch_size = prmb.shape[0]
    num_muscles = prmb.shape[1]
    
    # 使用 Warp kernel 计算 bias 和 gain
    with wp.ScopedDevice(wp_device):
        # 计算 bias: 被动力，取决于肌肉长度
        bias_flat = wp.zeros(batch_size * num_muscles, dtype=float, device=wp_device)
        wp.launch(
            muscle_bias_kernel,
            dim=batch_size * num_muscles,
            inputs=[
                length.reshape(-1),
                lengthrange_batched.reshape(-1, 2),
                acc0.reshape(-1),
                prmb.reshape(-1, 10),
                bias_flat,
            ],
            device=bias_flat.device,
        )
        
        # 计算 gain: 主动力增益，取决于肌肉长度和速度 (FLV 曲线)
        gain_flat = wp.zeros(batch_size * num_muscles, dtype=float, device=wp_device)
        wp.launch(
            muscle_gain_kernel,
            dim=batch_size * num_muscles,
            inputs=[
                length.reshape(-1),
                velocity.reshape(-1),
                lengthrange_batched.reshape(-1, 2),
                acc0.reshape(-1),
                prmg.reshape(-1, 10),
                gain_flat,
            ],
            device=gain_flat.device,
        )
        
        # 转换回 PyTorch Tensor 并 reshape
        bias = wp.to_torch(bias_flat).reshape(batch_size, num_muscles)
        gain = wp.to_torch(gain_flat).reshape(batch_size, num_muscles)
        
        # 确保 gain 为负值以保证数值稳定性（肌肉主动力为负）
        gain = torch.clamp(gain, max=-1.0)
        
        # 逆动力学解算: activation = (f - bias) / gain
        # 并将激活值限制在 [0, 1] 范围内
        activations = torch.clamp((f_clipped - bias) / gain, min=0.0, max=1.0)
    
    return activations
