from collections import OrderedDict
import numpy as np

import sapien.core as sapien
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

##################################################
# MS1
##################################################

from mani_skill2.envs.ms1.base_env import MS1BaseEnv

class MS1BaseEnv_fix(MS1BaseEnv):
    # The following code is to fix a bug in MS1 envs (0.5.3)
    def reset(self, seed=None, options=None):
        self._prev_actor_pose_dict = {}
        return super().reset(seed, options)
    
    def check_actor_static(self, actor: sapien.Actor, max_v=None, max_ang_v=None):
        """Check whether the actor is static by finite difference.
        Note that the angular velocity is normalized by pi due to legacy issues.
        """

        from mani_skill2.utils.geometry import angle_distance

        pose = actor.get_pose()

        if self._elapsed_steps <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            prev_actor_pose, prev_step, prev_actor_static = self._prev_actor_pose_dict[actor.id]
            if prev_step == self._elapsed_steps:
                return prev_actor_static
            assert prev_step == self._elapsed_steps - 1, (prev_step, self._elapsed_steps)
            dt = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - prev_actor_pose.p) <= max_v * dt
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(prev_actor_pose, pose) <= max_ang_v * dt
            )

        # CAUTION: carefully deal with it for MPC
        actor_static = flag_v and flag_ang_v
        self._prev_actor_pose_dict[actor.id] = (pose, self._elapsed_steps, actor_static)
        return actor_static

##################################################
# OpenCabinet
##################################################

from mani_skill2.envs.ms1.open_cabinet_door_drawer import (
    OpenCabinetEnv, 
    OpenCabinetDoorEnv,
    OpenCabinetDrawerEnv,
    clip_and_normalize,
)
import trimesh
from scipy.spatial import distance as sdist
from mani_skill2.utils.geometry import angle_distance, transform_points

class OpenCabinetEnv_unified(OpenCabinetEnv, MS1BaseEnv_fix):
    # unify the state obs for different objects

    def _compute_grasp_poses(self, mesh: trimesh.Trimesh, pose: sapien.Pose):
        # we didn't modify this function, just save one varible from this function
        mesh2: trimesh.Trimesh = mesh.copy()
        mesh2.apply_transform(pose.to_transformation_matrix())
        extents = mesh2.extents
        if extents[1] > extents[2]:  # horizontal handle
            closing = np.array([0, 0, 1])
        else:  # vertical handle
            closing = np.array([0, 1, 0])
        self.extents = extents # save this

        approaching = [1, 0, 0]
        grasp_poses = [
            self.agent.build_grasp_pose(approaching, closing, [0, 0, 0]),
            self.agent.build_grasp_pose(approaching, -closing, [0, 0, 0]),
        ]
        pose_inv = pose.inv()
        grasp_poses = [pose_inv * x for x in grasp_poses]

        return grasp_poses
    
    def compute_dense_reward(self, *args, info: dict, **kwargs):
        reward = 0.0

        # -------------------------------------------------------------------------- #
        # The end-effector should be close to the target pose
        # -------------------------------------------------------------------------- #
        handle_pose = self.target_link.pose
        ee_pose = self.agent.hand.pose

        # Position
        ee_coords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
        handle_pcd = transform_points(
            handle_pose.to_transformation_matrix(), self.target_handle_pcd
        )
        # trimesh.PointCloud(handle_pcd).show()
        disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)
        dist_ee_to_handle = disp_ee_to_handle.reshape(2, -1).min(-1)  # [2]
        reward_ee_to_handle = -dist_ee_to_handle.mean() * 2
        reward += reward_ee_to_handle

        # Encourage grasping the handle
        ee_center_at_world = ee_coords.mean(0)  # [10, 3]
        ee_center_at_handle = transform_points(
            handle_pose.inv().to_transformation_matrix(), ee_center_at_world
        )
        # self.ee_center_at_handle = ee_center_at_handle
        dist_ee_center_to_handle = self.target_handle_sdf.signed_distance(
            ee_center_at_handle
        )
        # print("SDF", dist_ee_center_to_handle)
        dist_ee_center_to_handle = dist_ee_center_to_handle.max()
        reward_ee_center_to_handle = (
            clip_and_normalize(dist_ee_center_to_handle, -0.01, 4e-3) - 1
        )
        reward += reward_ee_center_to_handle

        # pointer = trimesh.creation.icosphere(radius=0.02, color=(1, 0, 0))
        # trimesh.Scene([self.target_handle_mesh, trimesh.PointCloud(ee_center_at_handle)]).show()

        # Rotation
        target_grasp_poses = self.target_handles_grasp_poses[self.target_link_idx]
        target_grasp_poses = [handle_pose * x for x in target_grasp_poses]
        angles_ee_to_grasp_poses = [
            angle_distance(ee_pose, x) for x in target_grasp_poses
        ]
        ee_rot_reward = -min(angles_ee_to_grasp_poses) / np.pi * 3
        reward += ee_rot_reward

        # -------------------------------------------------------------------------- #
        # Stage reward
        # -------------------------------------------------------------------------- #
        coeff_qvel = 1.5  # joint velocity
        coeff_qpos = 0.5  # joint position distance
        stage_reward = -5 - (coeff_qvel + coeff_qpos)
        # Legacy version also abstract coeff_qvel + coeff_qpos.

        link_qpos = info["link_qpos"]
        link_qvel = self.link_qvel
        link_vel_norm = info["link_vel_norm"]
        link_ang_vel_norm = info["link_ang_vel_norm"]

        ee_close_to_handle = (
            dist_ee_to_handle.max() <= 0.01 and dist_ee_center_to_handle > 0
        )
        if ee_close_to_handle:
            stage_reward += 0.5

            # Distance between current and target joint positions
            # TODO(jigu): the lower bound 0 is problematic? should we use lower bound of joint limits?
            reward_qpos = (
                clip_and_normalize(link_qpos, 0, self.target_qpos) * coeff_qpos
            )
            reward += reward_qpos

            if not info["open_enough"]:
                # Encourage positive joint velocity to increase joint position
                reward_qvel = clip_and_normalize(link_qvel, -0.1, 0.5) * coeff_qvel
                reward += reward_qvel
            else:
                # Add coeff_qvel for smooth transition of stagess
                stage_reward += 2 + coeff_qvel
                reward_static = -(link_vel_norm + link_ang_vel_norm * 0.5)
                reward += reward_static

                # Legacy version uses static from info, which is incompatible with MPC.
                # if info["cabinet_static"]:
                if link_vel_norm <= 0.1 and link_ang_vel_norm <= 1:
                    stage_reward += 1

        # Update info
        info.update(ee_close_to_handle=ee_close_to_handle, stage_reward=stage_reward)

        ########################################################
        # Update extra info for unified state obs
        ########################################################
        info.update(handle_center=handle_pcd.mean(axis=0))

        reward += stage_reward
        return reward

    def step(self, action):
        self.step_action(action)
        self._elapsed_steps += 1

        info = self.get_info()
        reward = self.get_reward(action=action, info=info)
        self.info = info
        obs = self.get_obs()
        terminated = self.get_done(obs=obs, info=info)
        return obs, reward, terminated, False, info

    def _get_obs_priviledged(self):
        if self._elapsed_steps == 0:
            info = self.get_info()
            reward = self.compute_dense_reward(info=info)
            self.info = info
        else:
            info = self.info
        obs = OrderedDict()

        # robot info
        obs.update(
            fingers_pos=self.agent.get_ee_coords().flatten(),
            ee_pose=vectorize_pose(self.agent.hand.pose),
        )

        # object info
        obs.update(
            target_link_pose=vectorize_pose(self.target_link.pose),
            target_angle_to_go=clip_and_normalize(info["link_qpos"], 0, self.target_qpos),
            handle_direction=float(self.extents[1] > self.extents[2]),
            handle_center=info["handle_center"],
        )

        return obs


@register_env("OpenCabinetDoor_unified-v1", max_episode_steps=200)
class OpenCabinetDoorEnv_unified(OpenCabinetEnv_unified, OpenCabinetDoorEnv):
    pass

@register_env("OpenCabinetDrawer_unified-v1", max_episode_steps=200)
class OpenCabinetDrawerEnv_unified(OpenCabinetEnv_unified, OpenCabinetDrawerEnv):
    pass

##################################################
# MoveBucket
##################################################

from mani_skill2.envs.ms1.move_bucket import MoveBucketEnv

@register_env("MoveBucket_unified-v1", max_episode_steps=200)
class MoveBucketEnv_unified(MoveBucketEnv, MS1BaseEnv_fix):
    def _get_obs_priviledged(self):
        return OrderedDict(
            ball=self.balls[0].pose.p,
            bucket=vectorize_pose(self.bucket_body_link.pose),
        )

##################################################
# PushChair
##################################################

from mani_skill2.envs.ms1.push_chair import PushChairEnv

@register_env("PushChair_unified-v1", max_episode_steps=200)
class PushChairEnv_unified(PushChairEnv, MS1BaseEnv_fix):
    def _set_chair_links(self):
        chair_links = self.chair.get_links()

        # Infer link types
        self.root_link = chair_links[0]
        self.wheel_links = []
        self.seat_link = None
        self.support_link = None
        self.back_link = None
        for link in chair_links:
            link_types = self._check_link_types(link)
            if "wheel" in link_types:
                self.wheel_links.append(link)
            if "seat" in link_types:
                assert self.seat_link is None, (self.seat_link, link)
                self.seat_link = link
            if "support" in link_types:
                assert self.support_link is None, (self.support_link, link)
                self.support_link = link
            if "back" in link_types:
                assert self.back_link is None, (self.back_link, link)
                self.back_link = link

        # Set the physical material for wheels
        wheel_material = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0
        )
        for link in self.wheel_links:
            for s in link.get_collision_shapes():
                s.set_physical_material(wheel_material)

    @staticmethod
    def _check_link_types(link: sapien.LinkBase):
        link_types = []
        for visual_body in link.get_visual_bodies():
            name = visual_body.name
            if "wheel" in name:
                link_types.append("wheel")
            if "seat" in name:
                link_types.append("seat")
            if "leg" in name or "foot" in name:
                link_types.append("support")
            if "back" in name:
                link_types.append("back")
        return link_types
    
    def _get_obs_priviledged(self):
        return OrderedDict(
            seat=vectorize_pose(self.back_link.pose),
        )