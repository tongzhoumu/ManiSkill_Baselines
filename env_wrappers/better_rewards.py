import numpy as np
from mani_skill2.utils.registration import register_env
from sapien.core import Pose

##################################################
# PickCube
##################################################

from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv

@register_env("PickCube-v1", max_episode_steps=200)
class PickCubeEnv_v1(PickCubeEnv):
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj) # remove max_angle=30 yeilds much better performance
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

            # static reward
            if self.check_obj_placed():
                qvel = self.agent.robot.get_qvel()[:-2]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward
    
##################################################
# StackCube
##################################################

from mani_skill2.envs.pick_and_place.stack_cube import StackCubeEnv

@register_env("StackCube-v1", max_episode_steps=200)
class StackCubeEnv_v1(StackCubeEnv):
    def reaching_reward(self):
        # reaching object reward
        tcp_pose = self.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
        return 1 - np.tanh(5 * cubeA_to_tcp_dist)

    def place_reward(self):
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2])
        cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)
        reaching_reward2 = 1 - np.tanh(5.0 * cubeA_to_goal_dist)
        return reaching_reward2

    def ungrasp_reward(self):
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda
        # ungrasp reward
        is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
        if not is_cubeA_grasped:
            reward = 1.0
        else:
            reward = np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width

        v = np.linalg.norm(self.cubeA.velocity)
        av = np.linalg.norm(self.cubeA.angular_velocity)
        static_reward = 1 - np.tanh(v*10 + av)
        
        return (reward + static_reward) / 2.0
        
    def compute_dense_reward(self, info, **kwargs):

        if info["success"]:
            reward = 8
        elif self._check_cubeA_on_cubeB():
            reward = 6 + self.ungrasp_reward()
        elif self.agent.check_grasp(self.cubeA):
            reward = 4 + self.place_reward()
        else:
            reward = 2 + self.reaching_reward()

        return reward

##################################################
# PickSingle
##################################################

from mani_skill2.envs.pick_and_place.pick_single import (
    PickSingleYCBEnv,
    PickSingleEGADEnv,
)

@register_env("PickSingleYCB-v1", max_episode_steps=200)
class PickSingleYCBEnv_v1(PickSingleYCBEnv):
    def check_obj_placed(self):
        obj_to_goal_pos = self.goal_pos - self.obj_pose.p
        return np.linalg.norm(obj_to_goal_pos) <= self.goal_thresh

    def reaching_reward(self):
        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        return reaching_reward

    def place_reward(self):
        obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        return place_reward

    def static_reward(self):
        qvel = self.agent.robot.get_qvel()[:-2]
        static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
        return static_reward

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        reward += 1 + self.reaching_reward()
        if self.agent.check_grasp(self.obj):
            reward += 1 + self.place_reward()
            if self.check_obj_placed():
                reward += 1 + self.static_reward()
                if info["success"]:
                    reward += 1

        return reward
    
@register_env("PickSingleEGAD-v1", max_episode_steps=200, obj_init_rot=0.2)
class PickSingleEGADEnv_v1(PickSingleEGADEnv, PickSingleYCBEnv_v1):
    pass


##################################################
# PickClutter
##################################################

from mani_skill2.envs.pick_and_place.pick_clutter import PickClutterYCBEnv

@register_env("PickClutterYCB-v1", max_episode_steps=200)
class PickClutterYCBEnv_v1(PickClutterYCBEnv):
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 12.0
        else:
        
            obj_pose = self.obj_pose

            # reaching reward
            tcp_wrt_obj_pose = obj_pose.inv() * self.tcp.pose
            tcp_to_obj_dist = np.linalg.norm(tcp_wrt_obj_pose.p)
            reaching_reward = 3.0 - np.tanh(
                3.0*tcp_to_obj_dist
            )
            reward = reward + reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.obj, max_angle=60)
            reward += 3.0 if is_grasped else 0.0

            # reaching-goal reward
            if is_grasped:
                obj_to_goal_pos = self.goal_pos - obj_pose.p
                obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)
                reaching_goal_reward = 3.0*(1- np.tanh(3.0 * obj_to_goal_dist))
                reward += reaching_goal_reward

        return reward

##################################################
# PegInsertionSide
##################################################

from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv

@register_env("PegInsertionSide-v1", max_episode_steps=200)
class PegInsertionSideEnv_v1(PegInsertionSideEnv):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            peg_radius = self.peg_half_size[-1]
            peg_half_length = self.peg_half_size[0]

            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+peg_radius)/peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 - np.tanh(5.0 * abs(peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward * 2 + align_reward_y + align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward