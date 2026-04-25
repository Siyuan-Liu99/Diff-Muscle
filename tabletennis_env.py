import time
from tqdm import tqdm
from ml_collections import config_dict

import torch
import numpy as np
import mujoco
import mujoco.viewer
import mujoco_warp as mjw
from scipy.spatial.transform import Rotation as R

from rsl_rl.env import VecEnv
from mjlab.sim.sim import Simulation, SimulationCfg, MujocoCfg
# from mjlab.third_party.isaaclab.isaaclab.utils import math
from mjlab.utils.lab_api import math

from planner import compute_land, compute_hit_pos, compute_paddle_vel, compute_land_net, vec_to_quat, compute_paddle_pos
from muscle_utils import get_target_actuator_length, target_length_to_activations, calculate_vae_muscle_act


def recursive_immobilize(spec, temp_model, parent, remove_eqs=False, remove_actuators=False):
    removed_joint_ids = []
    for s in parent.sites:
        spec.delete(s)
    for j in parent.joints:
        removed_joint_ids.extend(temp_model.joint(j.name).qposadr)
        if remove_eqs:
            for e in spec.equalities:
                if e.type == mujoco.mjtEq.mjEQ_JOINT and (e.name1 == j.name or e.name2 == j.name):
                    spec.delete(e)
        if remove_actuators:
            for a in spec.actuators:
                if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.target == j.name:
                    spec.delete(a)
        spec.delete(j)
    for child in parent.bodies:
        removed_joint_ids.extend(
            recursive_immobilize(spec, temp_model, child, remove_eqs, remove_actuators)
        )
    return removed_joint_ids


def recursive_remove_contacts(parent, return_condition=None):
    if return_condition is not None and return_condition(parent):
        return
    for g in parent.geoms:
        g.contype=0
        g.conaffinity=0
    for child in parent.bodies:
        recursive_remove_contacts(child, return_condition)


def recursive_mirror(meshes_to_mirror, spec_copy, parent):
    parent.pos[1] *= -1
    parent.quat[[1, 3]] *= -1
    parent.name += "_mirrored"

    # 重命名 joints
    for j in parent.joints:
        if j.name:
            j.name += "_mirrored"

    for g in parent.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            spec_copy.delete(g)
            continue
        g.pos[1] *= -1
        g.quat[[1, 3]] *= -1
        g.name += "_mirrored"
        g.group = 1
        meshes_to_mirror.add(g.meshname)
        g.meshname += "_mirrored"
    for child in parent.bodies:
        if "ping_pong" in child.name:
            spec_copy.detach_body(child)
            continue
        recursive_mirror(meshes_to_mirror, spec_copy, child)


def tabletennis_p2_cfg():
    return config_dict.create(
        # simulation configs
        model_path="tabletennis.xml",
        num_envs=1024,
        eval_env=False,
        nconmax=50000,
        njmax=200,
        frame_skip=5,
        action_type="joint_pd",  # choose from joint_pd, muscle_pd, muscle_act, muscle_vae
        kp_scale=10.0,
        kd_scale=0.1,
        # muscle_vae specific configs
        kp_vae=10.0,
        kd_vae=1.0,
        # task configs
        max_episode_length=300,
        normalize_act=True,
        ball_qvel=True,
        paddle_mass_range=(0.10, 0.15),
        # paddle_mass_range=(0.05, 0.25),
        # -0.9634, -0.2302, 1.4041
        ball_xyz_range=config_dict.create(
            high=(-1.79, 0.7, 1.35),
            low=(-1.81, -0.7, 1.1),
        ),
        ball_friction_range=config_dict.create(
            high=(1.1, 0.006, 0.00003),
            low=(0.9, 0.004, 0.00001),
        ),
        ball_limited_range=config_dict.create(
            high=(2.5, 1.5, 2.5),
            low=(-2.0, -1.5, 0.795),
        ),
        obs_keys=[
            # "time",
            "pelvis_pos",
            "body_qpos",
            "body_qvel",
            "ball_pos",
            "ball_vel",
            "paddle_pos",
            "paddle_vel",
            "paddle_ori",
            "reach_err",
            "touching_info",
            "act",
            "target_pos",
            "target_vel",
            # "target_time",
        ],
        critic_obs_keys=[
            "time",
            "pelvis_pos",
            "body_qpos",
            "body_qvel",
            "ball_pos",
            "ball_vel",
            "paddle_pos",
            "paddle_vel",
            "paddle_ori",
            "reach_err",
            "touching_info",
            "act",
            "actuator_length",
            "actuator_velocity",
            "target_pos",
            "target_vel",
            "target_time",
            "paddle_mass",
            "ball_friction",
        ],
        weighted_reward_keys={
            "rel_pos_err": 4,
            "rel_quat_err": 4,
            "fin_open": 10,
            "paddle_pos_err": 20,
            "paddle_ori_err": 10,
            "hit_with_paddle": 100,
            "fall_opponent": 100,
            "fall_plane_dist": 100,
            "fall_hit_plane": 100,
            "net_penalty": -20,
            "act_reg": 0,
        },
        # domain randomization
        enable_domain_randomization=False,
        pos_jitter_range=0.02,
        vel_jitter_range=0.2,
        act_jitter_range=0.05,
        enable_action_randomization=True,
        action_range=0.05,
        act_range=0.1,
    )


class TableTennisWarpEnv(VecEnv):
    @staticmethod
    def _preprocess_spec(
        spec: mujoco.MjSpec,
        remove_body_collisions: bool = True,
        add_left_arm: bool = True,
    ) -> mujoco.MjSpec:
        """Preprocess the MuJoCo spec to:
        - Immobilize leg joints
        - Remove unnecessary body collisions
        - Optionally add mirrored left arm
        
        Args:
            spec: The MuJoCo spec to preprocess
            remove_body_collisions: Whether to remove body collisions
            add_left_arm: Whether to add mirrored left arm
            
        Returns:
            The preprocessed spec
        """
        for s in spec.sensors:
            if "pingpong" not in s.name and "paddle" not in s.name and "ball" not in s.name:
                spec.delete(s)
        # Compile a temporary model to get joint information
        temp_model = spec.compile()
        
        # Immobilize leg joints
        removed_ids = recursive_immobilize(spec, temp_model, spec.body("femur_l"), remove_eqs=True)
        removed_ids.extend(recursive_immobilize(spec, temp_model, spec.body("femur_r"), remove_eqs=True))


        for key in spec.keys:
            key.qpos = [j for i, j in enumerate(key.qpos) if i not in removed_ids]

        if remove_body_collisions:
            recursive_remove_contacts(spec.body("full_body"), return_condition=lambda b: "radius" in b.name)
        
        return spec
    
    def __init__(self, cfg: config_dict.ConfigDict, device: str = "cuda:0"):
        self.cfg = cfg
        # Load spec and preprocess it
        spec: mujoco.MjSpec = mujoco.MjSpec.from_file(cfg.model_path)
        # spec = self._preprocess_spec(spec, remove_body_collisions=True, add_left_arm=True)
        self.mj_model = spec.compile()
        self.num_envs = cfg.num_envs
        self.max_episode_length = cfg.max_episode_length
        self.device = torch.device(device)
        # sim_cfg = SimulationCfg(
        #     nconmax=cfg.nconmax,
        #     njmax=cfg.njmax,
        #     mujoco=MujocoCfg(integrator="euler"),
        # )
        sim_cfg = SimulationCfg(nconmax=100)
        self.sim = Simulation(num_envs=self.num_envs, cfg=sim_cfg, model=self.mj_model, device=device)
        # domain randomization
        self.sim.expand_model_fields(["body_mass", "geom_friction", "body_pos"])
        self.sim.create_graph()
        self.renderer = None

        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.touching_info = torch.zeros(self.num_envs, 6, device=device, dtype=torch.bool)
        # ball touching state: 0 - start, 1 - touching the paddle, 2 - leave the paddle, 3 - touching something after leaving the paddle
        self.touching_state = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        self.after_leaving_paddle = torch.zeros(self.num_envs, device=device, dtype=torch.int)
        # ball landing state: 0 - not landing, 1 - landing, 2 - leave the table
        self.landing_state = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        self.hit_with_paddle_count = torch.zeros(self.num_envs, device=device, dtype=torch.int)

        self.after_leaving_own = torch.zeros(self.num_envs, device=device, dtype=torch.int)

        self.target_pos = torch.zeros(self.num_envs, 3, device=device)
        self.target_vel = torch.zeros(self.num_envs, 3, device=device)
        self.target_ori = torch.zeros(self.num_envs, 4, device=device)
        self.hit_pos = torch.zeros(self.num_envs, 3, device=device)
        self.target_time = torch.zeros(self.num_envs, device=device, dtype=torch.float)
        self.target_time_tolerance = 0.05
        self.leaving_paddle_tolerance = 3
        self.leaving_table_tolerance = 1

        self.equalities = []
        self.constrained_joints = []
        self.action_joints = []

        self.last_obs = None
        self.current_obs = None
        self.extras = None

        self.current_episode_length = 0
        self.hit_step = torch.zeros(self.num_envs, 1, device=device)
        self.paddle_face_dir_local = torch.zeros(1, 3, device=device)
        self.paddle_face_dir_local[0,2] = -1

        self.opponent_table_upper = torch.tensor([-1.35, 0.50, 0.785]).to(self.device)
        self.opponent_table_lower = torch.tensor([-0.5, -0.40, 0.785]).to(self.device)

        for i in range(self.mj_model.neq):
            self.equalities.append([self.mj_model.eq_obj1id[i], self.mj_model.eq_obj2id[i], self.mj_model.eq_data[i]])
            self.constrained_joints.append(self.mj_model.eq_obj1id[i])

        for i in range(self.mj_model.njnt):
            if (
                self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
                or self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE
            ) and i not in self.constrained_joints:
                self.action_joints.append(i)

        if self.cfg.action_type == "joint_pd":
            self.num_actions = len(self.action_joints)
        elif self.cfg.action_type in ["muscle_pd", "muscle_act", "muscle_vae"]:
            self.num_actions = self.sim.data.ctrl.shape[1]

        self.action_joints = torch.tensor(self.action_joints).to(device)
        self.action_low = torch.from_numpy(self.mj_model.jnt_range[:, 0]).to(device)[self.action_joints].float()
        self.action_high = torch.from_numpy(self.mj_model.jnt_range[:, 1]).to(device)[self.action_joints].float()
        self.action_to_qpos = torch.from_numpy(self.mj_model.jnt_qposadr).to(device)[self.action_joints]

        self._post_init()

    @property
    def model(self) -> mjw.Model:
        return self.sim.model

    @property
    def data(self) -> mjw.Data:
        return self.sim.data

    @property
    def fk_data(self) -> mjw.Data:
        return self.sim.fk_data

    def _post_init(self):
        """record some index to calculate observations and rewards"""
        self.muscle_ind = torch.from_numpy(self.mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).to(self.device)
        self.non_muscle_ind = torch.from_numpy(self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE).to(
            self.device
        )
        self.non_muscle_low = (
            torch.from_numpy(
                self.mj_model.actuator_ctrlrange[self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE, 0]
            )
            .to(self.device)
            .float()
        )
        self.non_muscle_high = (
            torch.from_numpy(
                self.mj_model.actuator_ctrlrange[self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE, 1]
            )
            .to(self.device)
            .float()
        )

        self.init_qpos = torch.from_numpy(self.mj_model.key_qpos[0].copy()).to(self.device).float()

        self.ball_xyz_low = torch.tensor(self.cfg.ball_xyz_range.low).to(self.device).float()
        self.ball_xyz_high = torch.tensor(self.cfg.ball_xyz_range.high).to(self.device).float()
        self.ball_friction_low = torch.tensor(self.cfg.ball_friction_range.low).to(self.device).float()
        self.ball_friction_high = torch.tensor(self.cfg.ball_friction_range.high).to(self.device).float()
        self.paddle_mass_low = self.cfg.paddle_mass_range[0]
        self.paddle_mass_high = self.cfg.paddle_mass_range[1]
        self.ball_limited_low = torch.tensor(self.cfg.ball_limited_range.low).to(self.device).float()
        self.ball_limited_high = torch.tensor(self.cfg.ball_limited_range.high).to(self.device).float()

        self.opponent_center = torch.tensor([-0.85, 0.04, 0.795]).to(self.device)
        self.plane_center = (self.ball_xyz_low + self.ball_xyz_high) / 2

        self.palm_sid = self.mj_model.site("S_grasp").id
        self.fin0_sid = self.mj_model.site("THtip").id
        self.fin1_sid = self.mj_model.site("IFtip").id
        self.fin2_sid = self.mj_model.site("MFtip").id
        self.fin3_sid = self.mj_model.site("RFtip").id
        self.fin4_sid = self.mj_model.site("LFtip").id

        self.pelvis_sid = self.mj_model.site("pelvis").id
        self.paddle_sid = self.mj_model.site("paddle").id
        self.paddle_bid = self.mj_model.body("paddle").id
        self.ball_sid = self.mj_model.site("pingpong").id
        self.ball_bid = self.mj_model.body("pingpong").id
        self.grasp_sid = self.mj_model.site("S_grasp").id

        self.ball_bid = self.mj_model.body("pingpong").id
        self.ball_gid = self.mj_model.geom("pingpong").id
        self.own_half_gid = self.mj_model.geom("coll_own_half").id
        self.paddle_gid = self.mj_model.geom("pad").id
        self.opponent_half_gid = self.mj_model.geom("coll_opponent_half").id
        self.ground_gid = self.mj_model.geom("ground").id
        self.net_gid = self.mj_model.geom("coll_net").id

        self.ball_sensor_adr = self.mj_model.sensor("pingpong_vel_sensor").adr[0]
        self.ball_sensor_dim = self.mj_model.sensor("pingpong_vel_sensor").dim[0]
        self.paddle_sensor_adr = self.mj_model.sensor("paddle_vel_sensor").adr[0]
        self.paddle_sensor_dim = self.mj_model.sensor("paddle_vel_sensor").dim[0]

        self.ball_paddle_sensor_adr = self.mj_model.sensor("ball_paddle_contact").adr[0]
        self.ball_paddle_sensor_dim = self.mj_model.sensor("ball_paddle_contact").dim[0]
        self.ball_own_sensor_adr = self.mj_model.sensor("ball_own_contact").adr[0]
        self.ball_own_sensor_dim = self.mj_model.sensor("ball_own_contact").dim[0]
        self.ball_opponent_sensor_adr = self.mj_model.sensor("ball_opponent_contact").adr[0]
        self.ball_opponent_sensor_dim = self.mj_model.sensor("ball_opponent_contact").dim[0]
        self.ball_ground_sensor_adr = self.mj_model.sensor("ball_ground_contact").adr[0]
        self.ball_ground_sensor_dim = self.mj_model.sensor("ball_ground_contact").dim[0]
        self.ball_net_sensor_adr = self.mj_model.sensor("ball_net_contact").adr[0]
        self.ball_net_sensor_dim = self.mj_model.sensor("ball_net_contact").dim[0]
        self.ball_other_sensor_adr = self.mj_model.sensor("ball_other_contact").adr[0]
        self.ball_other_sensor_dim = self.mj_model.sensor("ball_other_contact").dim[0]

        self.ball_dofadr = self.mj_model.body_dofadr[self.ball_bid]
        self.ball_posadr = self.mj_model.joint("pingpong_freejoint").qposadr[0]
        self.paddle_dofadr = self.mj_model.joint("paddle_freejoint").dofadr[0]
        self.paddle_posadr = self.mj_model.joint("paddle_freejoint").qposadr[0]

        myo_bodies = [
            self.mj_model.body(i).id
            for i in range(self.mj_model.nbody)
            if not self.mj_model.body(i).name.startswith("ping")
            and "paddle" not in self.mj_model.body(i).name
            and not self.mj_model.body(i).name in ["pingpong"]
        ]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        self.myo_joint_range = np.concatenate(
            [
                self.mj_model.joint(i).qposadr
                for i in range(self.mj_model.njnt)
                if not self.mj_model.joint(i).name.startswith("ping")
                and not self.mj_model.joint(i).name == "pingpong_freejoint"
                and not self.mj_model.joint(i).name == "paddle_freejoint"
            ]
        )

        self.myo_dof_range = np.concatenate(
            [
                self.mj_model.joint(i).dofadr
                for i in range(self.mj_model.njnt)
                if not self.mj_model.joint(i).name.startswith("ping")
                and not self.mj_model.joint(i).name == "paddle_freejoint"
            ]
        )

        # 计算 t-pose 姿态下的肌肉长度（用于 muscle_vae 模式）
        if self.cfg.action_type == "muscle_vae":
            self._compute_tpose_muscle_length()

    def _compute_tpose_muscle_length(self):
        """计算 t-pose（初始关键帧姿态）下的肌肉长度，用于 muscle_vae 动作映射"""
        # 使用 fk_data 计算 t-pose 姿态下的肌肉长度
        # 将初始 qpos 设置到 fk_data 中
        self.fk_data.qpos[:] = self.init_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        # 执行正运动学计算
        self.sim.fk_forward()
        # 获取 t-pose 下的肌肉长度 [num_envs, num_actuators]
        # 取第一个环境的值作为参考（所有环境应该相同）
        self.tpose_muscle_length = self.fk_data.actuator_length[0].clone()
        # 只保留肌肉执行器的长度 [num_muscles]
        self.tpose_muscle_length_muscles = self.tpose_muscle_length[self.muscle_ind].clone()

    def _cal_touching_info(self) -> torch.Tensor:
        paddle_contact = (
            self.data.sensordata[
                :, self.ball_paddle_sensor_adr : self.ball_paddle_sensor_adr + self.ball_paddle_sensor_dim
            ]
            > 0
        )
        own_contact = (
            self.data.sensordata[:, self.ball_own_sensor_adr : self.ball_own_sensor_adr + self.ball_own_sensor_dim] > 0
        )
        opponent_contact = (
            self.data.sensordata[
                :, self.ball_opponent_sensor_adr : self.ball_opponent_sensor_adr + self.ball_opponent_sensor_dim
            ]
            > 0
        )
        ground_contact = (
            self.data.sensordata[
                :, self.ball_ground_sensor_adr : self.ball_ground_sensor_adr + self.ball_ground_sensor_dim
            ]
            > 0
        )
        net_contact = (
            self.data.sensordata[:, self.ball_net_sensor_adr : self.ball_net_sensor_adr + self.ball_net_sensor_dim] > 0
        )
        env_contact = (
            self.data.sensordata[
                :, self.ball_other_sensor_adr : self.ball_other_sensor_adr + self.ball_other_sensor_dim
            ]
            > 0
        )
        env_contact &= ~paddle_contact
        env_contact &= ~own_contact
        env_contact &= ~opponent_contact
        env_contact &= ~ground_contact
        env_contact &= ~net_contact
        self.touching_info = torch.cat(
            [paddle_contact, own_contact, opponent_contact, ground_contact, net_contact, env_contact], dim=-1
        )

        # touching_state==0 & paddle_contact -> touching_state=1 touching the paddle
        self.touching_state = torch.where(
            (self.touching_state == 0) & paddle_contact.squeeze(-1), 1, self.touching_state
        )
        # touching_state==1 & ~paddle_contact -> touching_state=2 leaving the paddle
        self.touching_state = torch.where(
            (self.touching_state == 1) & ~paddle_contact.squeeze(-1), 2, self.touching_state
        )

        # touching_state==2 & (paddle_contact | own_contact | ground_contact | net_contact | env_contact) -> touching_state=3 touching other things after leaving the paddle
        self.touching_state = torch.where(
            (self.touching_state == 2)
            & (paddle_contact | own_contact | ground_contact | net_contact | env_contact).squeeze(-1),
            3,
            self.touching_state,
        )

        # touching_state==2 & opponent_contact -> touching_state=4 touching the opponent side
        self.touching_state = torch.where(
            (self.touching_state == 2)
            & (opponent_contact).squeeze(-1),
            4,
            self.touching_state,
        )

        self.after_leaving_paddle[self.touching_state >= 2] += 1 # after leaving the paddle, count the number of timesteps

        # landing_state==0 & own_contact -> landing_state=1
        self.landing_state = torch.where((self.landing_state == 0) & own_contact.squeeze(-1), 1, self.landing_state)
        # landing_state==1 & ~own_contact -> landing_state=2
        self.landing_state = torch.where((self.landing_state == 1) & ~own_contact.squeeze(-1), 2, self.landing_state)

        self.after_leaving_own[self.landing_state == 1] = 0
        self.after_leaving_own[self.landing_state >= 2] += 1

    def _update_current_obs(self) -> None:
        sim_time = self.data.time
        pelvis_pos = self.data.site_xpos[:, self.pelvis_sid]
        noised_pelvis_pos = pelvis_pos + torch.randn_like(pelvis_pos) * self.cfg.pos_jitter_range

        body_qpos = self.data.qpos[:, self.myo_joint_range]
        body_qvel = self.data.qvel[:, self.myo_dof_range]
        noised_body_qpos = body_qpos + torch.randn_like(body_qpos) * self.cfg.pos_jitter_range
        noised_body_qvel = body_qvel + torch.randn_like(body_qvel) * self.cfg.vel_jitter_range

        ball_pos = self.data.site_xpos[:, self.ball_sid]
        ball_vel = self.data.sensordata[:, self.ball_sensor_adr : self.ball_sensor_adr + self.ball_sensor_dim]

        paddle_pos = self.data.site_xpos[:, self.paddle_sid]
        paddle_vel = self.data.sensordata[:, self.paddle_sensor_adr : self.paddle_sensor_adr + self.paddle_sensor_dim]
        paddle_ori = self.data.xquat[:, self.paddle_bid]
        noised_paddle_pos = paddle_pos + torch.randn_like(paddle_pos) * self.cfg.pos_jitter_range
        noised_paddle_vel = paddle_vel + torch.randn_like(paddle_vel) * self.cfg.vel_jitter_range

        reach_err = paddle_pos - ball_pos
        palm_pos = self.data.site_xpos[:, self.grasp_sid]
        palm_err = palm_pos - paddle_pos

        noised_palm_pos = palm_pos + torch.randn_like(palm_pos) * self.cfg.pos_jitter_range
        noised_reach_err = noised_paddle_pos - ball_pos
        noised_palm_err = noised_palm_pos - noised_paddle_pos

        target_pos = self.target_pos
        target_vel = self.target_vel
        target_time = self.target_time.unsqueeze(-1)

        act = self.data.act.clone()
        noised_act = act + torch.randn_like(act) * self.cfg.act_jitter_range

        actuator_length = self.data.actuator_length
        actuator_velocity = self.data.actuator_velocity

        critic_obs_dict = {
            "time": sim_time.unsqueeze(-1),
            "pelvis_pos": pelvis_pos,
            "body_qpos": body_qpos,
            "body_qvel": body_qvel,
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "paddle_pos": paddle_pos,
            "paddle_vel": paddle_vel,
            "paddle_ori": paddle_ori,
            "reach_err": reach_err,
            "palm_err": palm_err,
            "touching_info": self.touching_info,
            "act": act,
            "actuator_length": actuator_length,
            "actuator_velocity": actuator_velocity,
            "target_pos": target_pos,
            "target_vel": target_vel,
            "target_time": target_time,
            "paddle_mass": self.model.body_mass[:, self.paddle_bid].unsqueeze(-1),
            "ball_friction": self.model.geom_friction[:, self.ball_gid],
        }

        if self.cfg.enable_domain_randomization:
            obs_dict = {
                "time": sim_time.unsqueeze(-1),
                "pelvis_pos": noised_pelvis_pos,
                "body_qpos": noised_body_qpos,
                "body_qvel": noised_body_qvel,
                "ball_pos": ball_pos,
                "ball_vel": ball_vel,
                "paddle_pos": noised_paddle_pos,
                "paddle_vel": noised_paddle_vel,
                "paddle_ori": paddle_ori,
                "reach_err": noised_reach_err,
                "palm_err": noised_palm_err,
                "touching_info": self.touching_info,
                "act": noised_act,
                "target_pos": target_pos,
                "target_vel": target_vel,
                "target_time": target_time,
            }
        else:
            obs_dict = critic_obs_dict

        obs_list = list([obs_dict[k].clone() for k in self.cfg.obs_keys])
        critic_obs_list = list([critic_obs_dict[k].clone() for k in self.cfg.critic_obs_keys])

        # for key in self.cfg.obs_keys:
        #     print(key, obs_dict[key].shape)
        # exit()

        self.current_obs = torch.cat(obs_list, dim=-1).nan_to_num(0)
        self.extras = {
            "observations": {"critic": torch.cat(critic_obs_list, dim=-1).nan_to_num(0)},
            "obs_dict": obs_dict,
            "log": {},
            "time_outs": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # obs = torch.cat([self.last_obs, self.current_obs], dim=-1)

        obs = self.current_obs
        return obs, self.extras

    def _rand_ball_pos_and_vel(self, n_reset_envs: int):
        ball_qpos = (
            torch.rand((n_reset_envs, 3)).to(self.device) * (self.ball_xyz_high - self.ball_xyz_low) + self.ball_xyz_low
        )
        table_upper = torch.tensor([1.35, 0.50, 0.785]).to(self.device)
        table_lower = torch.tensor([0.5, -0.40, 0.785]).to(self.device)
        gravity = 9.81
        v_z = torch.rand((n_reset_envs,)).to(self.device) * 0.2 + 0.6

        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[:, 2] - table_upper[2]

        discriminant = b**2 - 4 * a * c
        t = (-b - discriminant**0.5) / (2 * a)

        if (discriminant < 0).any():
            raise ValueError(f"No real t: z0={ball_qpos[:, 2]}, z_target={table_upper[2]}, v_z_init={v_z}")

        v_upper = torch.stack(
            [(table_upper[0] - ball_qpos[:, 0]) / t, (table_upper[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        v_lower = torch.stack(
            [(table_lower[0] - ball_qpos[:, 0]) / t, (table_lower[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        ball_qvel = torch.rand((n_reset_envs, 3)).to(self.device) * (v_upper - v_lower) + v_lower

        return ball_qpos, ball_qvel

    def _rand_ball_vel(self, n_reset_envs: int, ball_qpos: torch.Tensor):

        table_upper = torch.tensor([1.35, 0.50, 0.785]).to(self.device)
        table_lower = torch.tensor([0.5, -0.40, 0.785]).to(self.device)
        gravity = 9.81
        v_z = torch.rand((n_reset_envs,)).to(self.device) * 0.2 + 0.4

        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[:, 2] - table_upper[2]

        discriminant = b**2 - 4 * a * c
        t = (-b - discriminant**0.5) / (2 * a)

        if (discriminant < 0).any():
            print(f"ball_qpos: {ball_qpos}")
            raise ValueError(f"No real t: z0={ball_qpos[:, 2]}, z_target={table_upper[2]}, v_z_init={v_z}")

        v_upper = torch.stack(
            [(table_upper[0] - ball_qpos[:, 0]) / t, (table_upper[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        v_lower = torch.stack(
            [(table_lower[0] - ball_qpos[:, 0]) / t, (table_lower[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        ball_qvel = torch.rand((n_reset_envs, 3)).to(self.device) * (v_upper - v_lower) + v_lower

        return ball_qvel



    def _get_termination_train(self) -> torch.Tensor:
        """Termination condition used for training, where the episode ends after 10s"""
        # the paddle did not touch the ball after hit time + tolerance
        ball_miss = (self.data.time > self.target_time + self.target_time_tolerance) & (self.touching_state == 0)

        # over max time limit
        max_time = self.data.time > 5

        # the position of ball is out of range
        ball_pos = self.data.site_xpos[:, self.ball_sid]
        ball_pos_in_range = (ball_pos >= self.ball_limited_low) & (ball_pos <= self.ball_limited_high)
        ball_pos_out_of_range = (~ball_pos_in_range).any(dim=-1)

        # the ball touched the net、own after leaving the paddle
        # touching_other_things = self.touching_state == 3
        # ball_finished = (self.touching_state == 2) & (self.after_leaving_paddle >= self.leaving_paddle_tolerance)

        # touching the opponent side
        # touching_opponent_side = self.touching_state == 4

        # leave the paddle
        leave_paddle = (self.touching_state == 2) & (self.after_leaving_paddle >= self.leaving_paddle_tolerance)

        # the ball touched something after leaving the paddle
        # ball_touched_after_leaving = self.touching_state == 3

        return max_time | ball_miss | ball_pos_out_of_range | leave_paddle

    def _get_termination_test(self) -> torch.Tensor:
        """Termination condition used for testing, where the episode ends after hitting the table"""
        # the paddle did not touch the ball after hit time + tolerance
        ball_miss = (self.data.time > self.target_time + self.target_time_tolerance) & (self.touching_state == 0)

        # over max time limit
        max_time = self.data.time > 5

        # the position of ball is out of range
        ball_pos = self.data.site_xpos[:, self.ball_sid]
        ball_pos_in_range = (ball_pos >= self.ball_limited_low) & (ball_pos <= self.ball_limited_high)
        ball_pos_out_of_range = (~ball_pos_in_range).any(dim=-1)

        # the ball touched the net、own after leaving the paddle
        # touching_other_things = self.touching_state == 3
        # ball_finished = (self.touching_state == 2) & (self.after_leaving_paddle >= self.leaving_paddle_tolerance)

        # touching the opponent side
        # touching_opponent_side = self.touching_state == 4

        # leave the paddle
        leave_paddle = (self.touching_state == 2) & (self.after_leaving_paddle >= self.leaving_paddle_tolerance)

        # the ball touched something after leaving the paddle
        # ball_touched_after_leaving = self.touching_state == 3

        return max_time | ball_miss | ball_pos_out_of_range

    def _cal_reward(self) -> tuple[torch.Tensor, dict]:
        """Calculate the reward"""

        # dense rewards
        rel_pos, rel_quat = self._cal_paddle_hand_rel_pose()
        rel_pos_err = torch.norm(rel_pos - self.init_rel_pos, dim=-1)
        rel_quat_err = 2 * torch.arccos(torch.clamp(torch.abs(torch.sum(rel_quat * self.init_rel_quat, dim=-1)), 0, 1))

        fin_open = self._get_fin_open()
        
        # if leave the paddle, paddle_ori_err and paddle_pos_err should be 0
        paddle_ori_err = self._get_paddle_ori_err()
        paddle_pos_err = self._get_paddle_pos_err()
        paddle_ori_err[self.touching_state >= 1] = 0
        paddle_pos_err[self.touching_state >= 1] = 0

        # sparse rewards
        self.hit_with_paddle_count += (self.touching_state >= 1).int()
        fall_plane_dist = torch.zeros(self.num_envs, device=self.device).float()
        fall_hit_plane = torch.zeros(self.num_envs, device=self.device).float()
        fall_dist = torch.zeros(self.num_envs, device=self.device).float()
        fall_opponent = torch.zeros(self.num_envs, device=self.device).float()
        paddle_vel_err = torch.zeros(self.num_envs, device=self.device).float()
        net_penalty = torch.zeros(self.num_envs, device=self.device).float()

        reward_vel_mask = (self.data.time - self.target_time).abs() < 0.005
        if reward_vel_mask.any():
            paddle_vel_err[reward_vel_mask] = torch.exp(
                -0.5
                * torch.norm(
                    self.data.qvel[reward_vel_mask, self.paddle_dofadr : self.paddle_dofadr + 3]
                    - self.target_vel[reward_vel_mask],
                    dim=-1,
                )
            )

        # leaving the paddle and reach tolerance, calculate fall opponent reward
        if ((self.touching_state == 2) & (self.after_leaving_paddle == self.leaving_paddle_tolerance)).any():
            ball_pos = self.data.qpos[:, self.ball_posadr : self.ball_posadr + 3].clone()
            ball_vel = self.data.qvel[:, self.ball_dofadr : self.ball_dofadr + 3].clone()

            # index mask of env ids to use planner to calculate fall opponent reward
            touching_mask = (self.touching_state == 2) & (self.after_leaving_paddle == self.leaving_paddle_tolerance)
            t_land, land_pos, land_vel = compute_land(ball_pos[touching_mask], ball_vel[touching_mask])

            on_opponent_half = (
                (land_pos[:, 1] > -0.68) & (land_pos[:, 1] < 0.76) & (land_pos[:, 0] > -1.34) & (land_pos[:, 0] < -0.05)
            )
            fail = compute_land_net(ball_pos[touching_mask], ball_vel[touching_mask], net_h=0.95+0.05)
            success = on_opponent_half & ~fail
            net_penalty[touching_mask] = fail.float()   # 失败的就变为1

            # TODO: 如果成功落到对手桌面，计算在对手平面的落点
            bounce_vel = land_vel.clone()
            bounce_vel[:, 2] = -bounce_vel[:, 2]
            hit_plane_pos, _, _ = compute_hit_pos(t_land[success], land_pos[success], bounce_vel[success], hit_plane_x=-1.8)
            on_hit_plane = (hit_plane_pos[:,1] > -0.7) & (hit_plane_pos[:,1] < 0.7) & (hit_plane_pos[:,2] > 1.1) & (hit_plane_pos[:,2] < 1.35)

            touching_idx = torch.nonzero(touching_mask, as_tuple=True)[0]
            success_idx = torch.nonzero(success, as_tuple=True)[0]
            on_hit_plane_idx = torch.nonzero(on_hit_plane, as_tuple=True)[0]
            if len(on_hit_plane_idx) > 0:
                # breakpoint()
                fall_hit_plane[touching_idx[success_idx[on_hit_plane_idx]]] = on_hit_plane[on_hit_plane_idx].float()
            
            fall_opponent[touching_mask] = success.float()
            fall_plane_dist[touching_idx[success]] = torch.exp(
                -0.5 * torch.norm(hit_plane_pos[:,1:] - self.plane_center[1:], dim=-1)
            )


        act_reg = torch.norm(self.data.act, dim=-1) / self.model.na

        reward_dict = {
            "rel_pos_err": torch.exp(-20.0 * rel_pos_err),
            "rel_quat_err": torch.exp(-2.0 * rel_quat_err),
            "fin_open": torch.exp(-5.0 * fin_open),
            "paddle_pos_err": torch.exp(-8.0 * paddle_pos_err),
            "paddle_ori_err": torch.exp(-2.0 * paddle_ori_err),
            "hit_with_paddle": (self.hit_with_paddle_count == 1).float(),
            "fall_hit_plane": fall_hit_plane,
            "paddle_vel_err": paddle_vel_err,
            "fall_opponent": fall_opponent,
            "fall_plane_dist": fall_plane_dist,
            "net_penalty": net_penalty,
            "act_reg": act_reg,
        }

        reward = torch.sum(
            torch.stack(
                [reward_dict[k] * self.cfg.weighted_reward_keys[k] for k in self.cfg.weighted_reward_keys.keys()],
                dim=-1,
            ),
            dim=-1,
        )

        return reward.nan_to_num(0), reward_dict

    def _cal_paddle_hand_rel_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the relative position and orientation between paddle and hand"""
        paddle_pos = self.data.site_xpos[:, self.paddle_sid]
        paddle_ori = self.data.site_xmat[:, self.paddle_sid].reshape(-1, 3, 3)
        hand_pos = self.data.site_xpos[:, self.grasp_sid]
        hand_ori = self.data.site_xmat[:, self.grasp_sid].reshape(-1, 3, 3)

        paddle_hand_rel_ori = torch.bmm(hand_ori.transpose(1, 2), paddle_ori)
        paddle_hand_rel_quat = math.quat_from_matrix(paddle_hand_rel_ori)
        paddle_hand_rel_pos = torch.bmm(hand_ori.transpose(1, 2), (paddle_pos - hand_pos).unsqueeze(-1)).squeeze(-1)

        return paddle_hand_rel_pos, paddle_hand_rel_quat

    def _get_fin_open(self) -> torch.Tensor:
        palm_pos = self.data.site_xpos[:, self.palm_sid]
        fin0_err = torch.norm(self.data.site_xpos[:, self.fin0_sid] - palm_pos, dim=-1)
        fin1_err = torch.norm(self.data.site_xpos[:, self.fin1_sid] - palm_pos, dim=-1)
        fin2_err = torch.norm(self.data.site_xpos[:, self.fin2_sid] - palm_pos, dim=-1)
        fin3_err = torch.norm(self.data.site_xpos[:, self.fin3_sid] - palm_pos, dim=-1)
        fin4_err = torch.norm(self.data.site_xpos[:, self.fin4_sid] - palm_pos, dim=-1)
        fin_open = fin0_err + fin1_err + fin2_err + fin3_err + fin4_err

        return fin_open

    def _get_paddle_ori_err(self) -> torch.Tensor:
        paddle_face_dir = torch.bmm(
            self.data.site_xmat[:, self.paddle_sid].reshape(-1, 3, 3), self.init_paddle_face_dir.unsqueeze(-1)
        ).squeeze(-1)
        # 不分正反手，哪一面打球都可以
        paddle_ori_err = torch.arccos(
            torch.clamp(
                torch.abs(
                    torch.sum(paddle_face_dir * self.target_vel, dim=-1) / (torch.norm(self.target_vel, dim=-1) + 1e-6)
                ),
                0,
                1,
            )
        )

        return paddle_ori_err

    def _get_paddle_pos_err(self) -> torch.Tensor:
        # paddle site的位置在拍面的正中心
        target_vel_dir = self.target_vel / (torch.norm(self.target_vel, dim=-1, keepdim=True) + 1e-6)
        # 拍子厚度0.02，球的半径0.02，所以需要朝自身移动0.04
        real_target_pos = torch.zeros_like(self.target_pos)
        vel_to_opponent_mask = (target_vel_dir[:, 0] < 0)
        real_target_pos[vel_to_opponent_mask] = self.target_pos[vel_to_opponent_mask] - target_vel_dir[vel_to_opponent_mask] * 0.04
        real_target_pos[~vel_to_opponent_mask] = self.target_pos[~vel_to_opponent_mask] + target_vel_dir[~vel_to_opponent_mask] * 0.04

        # paddle_center_pos = self.data.site_xpos[:, self.paddle_sid]
        # 奖励paddle body的位置
        paddle_pos = self.data.qpos[:, self.paddle_posadr : self.paddle_posadr + 3]
        paddle_pos_err = torch.norm(paddle_pos - real_target_pos, dim=-1)

        return paddle_pos_err

    
    def _check_ball_cross_net(self, ball_qpos: torch.Tensor, ball_qvel: torch.Tensor) -> torch.Tensor:
        """Check if the ball crosses the net"""
        t_net = (0.0 - ball_qpos[:, 0]) / ball_qvel[:, 0]
        h_net = ball_qpos[:, 2] + ball_qvel[:, 2] * t_net + 0.5 * -9.81 * t_net**2
        cross_net_flag = (h_net > 0.98)
        return cross_net_flag


    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environment, resample the domain randomization parameters"""
        
        n_reset_envs = env_ids.shape[0]

        # clear episode info
        self.episode_length_buf[env_ids] = 0
        self.touching_state[env_ids] = 0
        self.after_leaving_paddle[env_ids] = 0
        self.landing_state[env_ids] = 0
        self.hit_with_paddle_count[env_ids] = 0
        self.touching_info[env_ids] = 0

        self.after_leaving_own[env_ids] = 0

        # domain randomization for mjmodel
        # self.model.body_mass[env_ids, self.paddle_bid] = (
        #     torch.rand((n_reset_envs,)).to(self.device) * (self.paddle_mass_high - self.paddle_mass_low)
        #     + self.paddle_mass_low
        # )
        # self.model.geom_friction[env_ids, self.ball_gid] = (
        #     torch.rand((n_reset_envs, 3)).to(self.device) * (self.ball_friction_high - self.ball_friction_low)
        #     + self.ball_friction_low
        # )
        # self.model.body_mass[env_ids, self.paddle_bid] = 0.1318480843660727
        # self.model.geom_friction[env_ids, self.ball_gid] = torch.tensor([9.5396e-01, 4.0819e-03, 1.0331e-05]).to(self.device)

        # randomization on ball position, calculate every reset
        init_ball_qpos = torch.zeros(n_reset_envs, 3, device=self.device)
        init_ball_qvel = torch.zeros(n_reset_envs, 3, device=self.device)

        # TODO: 检测球网，并重新采样
        cross_net_flag = torch.zeros(n_reset_envs, device=self.device, dtype=torch.bool)
        while not cross_net_flag.all().item():
            n_envs_remain = (~cross_net_flag).sum().item()
            init_ball_qpos_remain, init_ball_qvel_remain = self._rand_ball_pos_and_vel(n_envs_remain)
            cross_net_flag_remain = self._check_ball_cross_net(init_ball_qpos_remain, init_ball_qvel_remain)

            # TODO: 判断发球后的target pos是否在范围内,满足这个范围才发球
            init_ball_qpos_judge = init_ball_qpos_remain.clone()
            init_ball_qvel_judge = init_ball_qvel_remain.clone()

            # TODO: 仅仅需要算hit_pos
            t_land, land_pos, land_vel = compute_land(init_ball_qpos_judge, init_ball_qvel_judge)
            bounce_vel = land_vel.clone()
            bounce_vel[:, 2] = -bounce_vel[:, 2]
            hit_pos, _, _ = compute_hit_pos(t_land, land_pos, bounce_vel)

            target_in_range = (hit_pos[:, 1] > self.ball_xyz_low[1]) & (hit_pos[:, 1] < self.ball_xyz_high[1]) & (hit_pos[:, 2] > self.ball_xyz_low[2]) & (hit_pos[:, 2] < self.ball_xyz_high[2])
            
            cross_net_flag_remain = cross_net_flag_remain & target_in_range


            n_success = cross_net_flag_remain.sum().item()

            fail_indices = torch.where(~cross_net_flag)[0]
            init_ball_qpos[fail_indices[:n_success]] = init_ball_qpos_remain[cross_net_flag_remain]
            init_ball_qvel[fail_indices[:n_success]] = init_ball_qvel_remain[cross_net_flag_remain]
            cross_net_flag[fail_indices[:n_success]] = True

            # print(f"n_envs_remain: {n_envs_remain}, n_success: {n_success}")

        # local_env_ids = torch.arange(n_reset_envs, device=self.device)
        # init_ball_qpos[local_env_ids] = torch.tensor([-0.9634, -0.2302, 1.4041]).to(self.device)
        # init_ball_qvel[local_env_ids] = torch.tensor([6.2354, 2.3637, -0.0967]).to(self.device)
        # self.model.body_pos[env_ids, self.ball_bid] = init_ball_qpos

        self.data.time[env_ids] = 0.0
        self.data.qpos[env_ids] = self.init_qpos
        self.data.qpos[env_ids, self.ball_posadr : self.ball_posadr + 3] = init_ball_qpos
        self.data.qvel[env_ids] = 0.0
        self.data.qvel[env_ids, self.ball_dofadr : self.ball_dofadr + 3] = init_ball_qvel
        self.data.qfrc_applied[env_ids] = 0
        self.data.xfrc_applied[env_ids] = 0
        self.data.ctrl[env_ids] = 0.0
        self.data.act[env_ids] = 0.0
        self.sim.forward()

        # get high command
        paddle_pos, paddle_vel, paddle_ori, hit_time, hit_pos = self.get_high_command(init_ball_qpos, init_ball_qvel)    

        self.target_pos[env_ids] = paddle_pos
        self.target_vel[env_ids] = paddle_vel
        self.target_ori[env_ids] = paddle_ori
        self.hit_pos[env_ids] = hit_pos
        self.target_time[env_ids] = hit_time

    
    def get_high_command(self, init_ball_qpos, init_ball_qvel, t_land=None):
        if t_land is None: # from initial position
            t_land, land_pos, land_vel = compute_land(init_ball_qpos, init_ball_qvel)
            bounce_vel = land_vel.clone()
            bounce_vel[:, 2] = -bounce_vel[:, 2]
        else: # from landing position
            land_pos = init_ball_qpos
            land_vel = init_ball_qvel
            bounce_vel = land_vel.clone()

        hit_pos, v_in, hit_time = compute_hit_pos(t_land, land_pos, bounce_vel)
        paddle_vel = compute_paddle_vel(hit_pos, v_in, self.opponent_table_upper, self.opponent_table_lower, self.device)

        # if paddle is moving backward, reverse the velocity
        paddle_vel_for_ori = paddle_vel.clone()
        if (paddle_vel_for_ori[:, 0] > 0).any():
            back_mask = paddle_vel_for_ori[:, 0] > 0
            paddle_vel_for_ori[back_mask] = -paddle_vel_for_ori[back_mask]

        paddle_face_dir_local = self.paddle_face_dir_local.expand(paddle_vel_for_ori.shape[0], -1)
        paddle_ori = vec_to_quat(paddle_face_dir_local, paddle_vel_for_ori)

        paddle_pos = compute_paddle_pos(paddle_ori, hit_pos)

        return paddle_pos, paddle_vel, paddle_ori, hit_time, hit_pos


    def reset(self) -> tuple[torch.Tensor, dict]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_rel_pos, self.init_rel_quat = self._cal_paddle_hand_rel_pose()
        self.ref_face_dir = torch.tensor([-1, 0, 0]).float().to(self.device)
        self.init_paddle_face_dir = torch.bmm(
            self.data.site_xmat[:, self.paddle_sid].reshape(-1, 3, 3).transpose(1, 2),
            self.ref_face_dir.unsqueeze(0).unsqueeze(-1).repeat_interleave(self.num_envs, dim=0),
        ).squeeze(-1)
        self._update_current_obs()
        self.last_obs = self.current_obs.clone()
        return self.get_observations()

    def joint_pd_to_muscle_act(self, actions: torch.Tensor) -> torch.Tensor:
        # TODO： 策略的前两维控制pelvis，后273维控制肌肉
        # add perturbation to actions
        if self.cfg.enable_action_randomization:
            actions_perturbation = torch.randn_like(actions) * self.cfg.action_range
            actions += actions_perturbation

        actions = torch.clamp(actions, -1, 1)
        pelvis_actions = actions[:, :2].clone()
        actions = (actions + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        target_length = get_target_actuator_length(
            self.model, self.fk_data, actions, self.equalities, self.action_to_qpos, self.sim.fk_forward
        )

        self.target_length = target_length.clone()

        kp_scale = self.cfg.kp_scale
        kd_scale = self.cfg.kd_scale

        activations, bias, gain, clipped_force = target_length_to_activations(
            self.model, self.data, target_length, kp_scale, kd_scale
        )

        self.bias = bias.clone()
        self.gain = gain.clone()
        self.clipped_force = clipped_force.clone()

        if self.cfg.enable_action_randomization:
            activation_perturbation = torch.randn_like(activations) * self.cfg.act_range
            activations += activation_perturbation

        activations = torch.clamp(
            activations,
            1.0 / (1.0 + torch.exp(torch.tensor(7.5, device=activations.device))),
            1.0 / (1.0 + torch.exp(torch.tensor(-2.5, device=activations.device))),
        )

        # [-1, 1] -> [-1, 0.05] (not joint range [-1, -0.05])
        pelvis_actions = (pelvis_actions + 1) / 2 * (self.non_muscle_high - self.non_muscle_low) + self.non_muscle_low

        return torch.cat([activations, pelvis_actions], dim=-1)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.clone()

        if self.cfg.action_type == "joint_pd":
            self.data.ctrl[:] = self.joint_pd_to_muscle_act(actions)
        elif self.cfg.action_type == "muscle_vae":
            # ========== muscle_vae 模式 ==========
            # Step 1: 动作解析 - 将策略输出 a ∈ [-1, 1] 映射为目标肌肉长度
            # 公式: target_length = (a + 1.0) * l_M^{tpose}
            # 这里 a=0 对应 t-pose 长度，a=-1 对应 0，a=1 对应 2*tpose
            
            # 分离肌肉动作和 pelvis 动作
            muscle_actions = actions[:, self.muscle_ind]  # [batch, 273]
            pelvis_actions = actions[:, self.non_muscle_ind]  # [batch, 2]
            
            # 映射到目标肌肉长度: target_length = (a + 1.0) * tpose_length
            target_length_muscles = (muscle_actions + 1.0) * self.tpose_muscle_length_muscles.unsqueeze(0)
            
            # 构建完整的 target_length tensor [batch, num_actuators]
            target_length = torch.zeros(actions.shape[0], self.data.actuator_length.shape[1], device=self.device)
            target_length[:, self.muscle_ind] = target_length_muscles
            
            # Step 2: 调用 VAE 控制器计算肌肉激活
            muscle_activations = calculate_vae_muscle_act(
                model=self.model,
                data=self.data,
                target_length=target_length,
                kp=self.cfg.kp_vae,
                kd=self.cfg.kd_vae,
            )
            
            # Step 3: 处理 Pelvis - 线性映射到控制范围
            pelvis_ctrl = (pelvis_actions + 1) / 2 * (
                self.non_muscle_high - self.non_muscle_low
            ) + self.non_muscle_low
            
            # Step 4: 组装完整的 275 维控制信号并写入
            ctrl = torch.zeros_like(self.data.ctrl)
            ctrl[:, self.muscle_ind] = muscle_activations
            ctrl[:, self.non_muscle_ind] = pelvis_ctrl
            self.data.ctrl[:] = ctrl
        else:
            # muscle_pd 或 muscle_act 模式
            if self.cfg.normalize_act:
                actions[:, self.muscle_ind] = 1.0 / (
                    1.0 + torch.exp(-5.0 * (actions[:, self.muscle_ind] - 0.5)).to(self.device)
                )
                actions[:, self.non_muscle_ind] = (actions[:, self.non_muscle_ind] + 1) / 2 * (
                    self.non_muscle_high - self.non_muscle_low
                ) + self.non_muscle_low
            self.data.ctrl[:] = actions

        for _ in range(self.cfg.frame_skip):
            self.sim.step()

        
        self._cal_touching_info()
        self.episode_length_buf += 1


        reward, reward_dict = self._cal_reward()
        if self.cfg.eval_env:
            done = self._get_termination_test()
        else:
            done = self._get_termination_train()
        if done.any():
            self._reset_idx(done.nonzero().squeeze(-1))

        # update last obs
        self.last_obs = self.current_obs.clone()
        self._update_current_obs()
        # handle reset envs
        self.last_obs[done] = self.current_obs[done].clone()
        obs, extras = self.get_observations()


        # recalculate own hit pos after landing own
        if ((self.landing_state == 2) & (self.after_leaving_own == self.leaving_table_tolerance)).any():
            landing_mask = (self.landing_state == 2) & (self.after_leaving_own == self.leaving_table_tolerance)
            ball_pos_mask = self.data.qpos[landing_mask, self.ball_posadr : self.ball_posadr + 3].clone()
            ball_vel_mask = self.data.qvel[landing_mask, self.ball_dofadr : self.ball_dofadr + 3].clone()

            paddle_pos, paddle_vel, paddle_ori, hit_time, hit_pos =self.get_high_command(ball_pos_mask, ball_vel_mask, self.data.time[landing_mask])

            self.target_pos[landing_mask] = paddle_pos
            self.target_vel[landing_mask] = paddle_vel
            self.target_ori[landing_mask] = paddle_ori
            self.hit_pos[landing_mask] = hit_pos
            self.target_time[landing_mask] = hit_time
        

        for key in reward_dict.keys():
            extras["log"]["reward/" + key] = reward_dict[key]

        return obs, reward, done, extras

    def render_offscreen(self):
        height = 480
        width = 640
        camera = 1
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.sim.mj_model, height=height, width=width
            )
            self.renderer.scene.ngeom += 1
        mjw.get_data_into(self.sim.mj_data, self.sim.mj_model, self.sim.wp_data)

        self.renderer.update_scene(self.sim.mj_data, camera=camera)
        # mujoco.mjv_initGeom(
        #     self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
        #     mujoco.mjtGeom.mjGEOM_SPHERE,
        #     np.ones(3) * 0.02,
        #     self.target_pos.cpu().numpy()[0],
        #     np.eye(3).flatten(),
        #     np.array([1.0, 0.0, 0.0, 0.7]),
        # )
        return self.renderer.render()


if __name__ == "__main__":
    cfg = tabletennis_p2_cfg()
    env = TableTennisWarpEnv(cfg)
    obs, info = env.reset()


    for _ in tqdm(range(1000)):
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, reward, done, info = env.step(actions)

    viewer = mujoco.viewer.launch_passive(env.sim.mj_model, env.sim.mj_data)
    viewer.user_scn.ngeom += 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.ones(3) * 0.02,
        env.target_pos.cpu().numpy()[0],
        np.eye(3).flatten(),
        np.array([1.0, 0.0, 0.0, 0.7]),
    )

    while viewer.is_running():
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, reward, done, info = env.step(actions)
        mjw.get_data_into(env.sim.mj_data, env.sim.mj_model, env.sim.wp_data)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.ones(3) * 0.02,
            env.target_pos.cpu().numpy()[0],
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 0.7]),
        )
        viewer.sync()
        time.sleep(0.01)
        if info["obs_dict"]["touching_info"][0].any():
            print(info["obs_dict"]["touching_info"][0])
            time.sleep(1.0)
