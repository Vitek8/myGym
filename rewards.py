import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import results_plotter
import os
import pybullet

class Reward:
    """
    Reward base class for reward signal calculation and visualization

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task=None):
        self.env = env
        self.task = task
        self.rewards_history = []

    def compute(self, observation=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def visualize_reward_over_steps(self):
        """
        Plot and save a graph of reward values assigned to individual steps during an episode. Call this method after the end of the episode.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_steps > 0:
            results_plotter.EPISODES_WINDOW=50
            results_plotter.plot_curves([(np.arange(self.env.episode_steps),np.asarray(self.rewards_history[-self.env.episode_steps:]))],'step','Step rewards')
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_steps_episode{}.png".format(self.env.episode_number))
            plt.close()

    def visualize_reward_over_episodes(self):
        """
        Plot and save a graph of cumulative reward values assigned to individual episodes. Call this method to plot data from the current and all previous episodes.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_number > 0:
            results_plotter.EPISODES_WINDOW=10
            results_plotter.plot_curves([(np.arange(self.env.episode_number),np.asarray(self.env.episode_final_reward[-self.env.episode_number:]))],'episode','Episode rewards')
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_episodes_episode{}.png".format(self.env.episode_number))
            plt.close()
        
class DistanceReward(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects
    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DistanceReward, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None


    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3])/2)]
        o2 = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3])/2):-3]
        reward = self.calc_dist_diff(o1, o2)
        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        return reward

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_obj1_position = None
        self.prev_obj2_position = None

    def calc_dist_diff(self, obj1_position, obj2_position):
        """
        Calculate change in the distance between 2 objects in previous and in current step. Normalize the change by the value of distance in previous step.
        Params:
            :param obj1_position: (list) Position of the first object
            :param obj2_position: (list) Position of the second object
        Returns:
            :return norm_diff: (float) Normalized difference of distances between 2 objects in previsous and in current step
        """
        if self.prev_obj1_position is None and self.prev_obj2_position is None:
            self.prev_obj1_position = obj1_position
            self.prev_obj2_position = obj2_position
        self.prev_diff = self.task.calc_distance(self.prev_obj1_position, self.prev_obj2_position)

        current_diff = self.task.calc_distance(obj1_position, obj2_position)
        norm_diff = (self.prev_diff - current_diff) / self.prev_diff

        self.prev_obj1_position = obj1_position
        self.prev_obj2_position = obj2_position

        return norm_diff
        
class ComplexDistanceReward(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 3 objects, e.g. 2 objects and gripper for complex tasks

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(ComplexDistanceReward,self).__init__(env, task)
        self.prev_obj3_position = None

    def compute(self, observation):
        """
        Compute reward signal based on distances between 3 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        reward = self.calc_dist_diff(observation[0:3], observation[3:6], observation[6:9])
        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        return reward

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        super().reset()
        self.prev_obj3_position = None

    def calc_dist_diff(self, obj1_position, obj2_position, obj3_position):
        """
        Calculate change in the distances between 3 objects in previous and in current step. Normalize the change by the value of distance in previous step.

        Params:
            :param obj1_position: (list) Position of the first object
            :param obj2_position: (list) Position of the second object
            :param obj3_position: (list) Position of the third object
        Returns:
            :return norm_diff: (float) Sum of normalized differences of distances between 3 objects in previsous and in current step
        """
        if self.prev_obj1_position is None and self.prev_obj2_position is None and self.prev_obj3_position is None:
            self.prev_obj1_position = obj1_position
            self.prev_obj2_position = obj2_position
            self.prev_obj3_position = obj3_position

        prev_diff_12 = self.task.calc_distance(self.prev_obj1_position, self.prev_obj2_position)
        current_diff_12 = self.task.calc_distance(obj1_position, obj2_position)

        prev_diff_13 = self.task.calc_distance(self.prev_obj1_position, self.prev_obj3_position)
        current_diff_13 = self.task.calc_distance(obj1_position, obj3_position)

        prev_diff_23 = self.task.calc_distance(self.prev_obj2_position, self.prev_obj3_position)
        current_diff_23 = self.task.calc_distance(obj2_position, obj3_position)
        
        norm_diff = (prev_diff_13 - current_diff_13) / prev_diff_13 + (prev_diff_23 - current_diff_23) / prev_diff_23 + (prev_diff_12 - current_diff_12) / prev_diff_12

        self.prev_obj1_position = obj1_position
        self.prev_obj2_position = obj2_position
        self.prev_obj3_position = obj3_position

        return norm_diff

class DistractorReward(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DistractorReward, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None

        self.prev_robot = None
        self.prev_distractor = None

        self.last_distances = []

        self.penalty = 0

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        # exit(observation)
        cube = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3])/2)]
        bus = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3])/2):-3]
        kuka = observation[6:9] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3])/2):-3]
        
        reward = self.calc_dist_diff(cube, kuka)
        reward += self.calc_links(observation[9:], bus)
        # reward -= self.calc_penalty(kuka, bus)
        
        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid, self.env.env_objects[0].uid, x, -1) for x in range(0,self.env.robot.end_effector_index+1)]
        for p in contact_points:
            if p:
                self.env.distractor_movable = False
                reward+=1
                break

        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid, self.env.env_objects[-1].uid, x, -1) for x in range(0,self.env.robot.end_effector_index+1)]
        for p in contact_points:
            if p:
                r = abs(self.calc_penalty(kuka, bus))
                reward -= r
                break

        # if kuka under table penalize + fail

        self.task.check_distance_threshold(observation=observation)

        self.rewards_history.append(reward)
        return reward

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_obj1_position = None
        self.prev_obj2_position = None

        self.prev_robot = None
        self.prev_distractor = None

        self.penalty = 0
        self.env.distractor_movable = True

        self.last_distances = [None]*11

    def calc_penalty(self, robot, distractor):
        """
        Calculate change in the distance between 2 objects in previous and in current step. Normalize the change by the value of distance in previous step.

        Params:
            :param obj1_position: (list) Position of the first object
            :param obj2_position: (list) Position of the second object
        Returns:
            :return norm_diff: (float) Normalized difference of distances between 2 objects in previsous and in current step
        """

        if self.prev_robot is None and self.prev_distractor is None:
            self.prev_robot = robot
            self.prev_distractor = distractor
        self.prev_diff = self.task.calc_distance(self.prev_robot, self.prev_distractor)

        current_diff = self.task.calc_distance(robot, distractor)
        norm_diff = (self.prev_diff - current_diff) / self.prev_diff

        self.prev_robot = robot
        self.prev_distractor = distractor
        self.penalty += 1

        if self.penalty > 19:
            self.env.episode_failed = True
            self.env.episode_over = True
            return 1

        return norm_diff

    def calc_links(self, links, distractor):
        reward = 0.3
        # rewards = []
        distances = self.last_distances
        i = 0
        j = 0

        for item in range(11):
            dist = self.task.calc_distance(links[i:i+3], distractor)
            if distances[j]:
                # reward += (distances[j] - dist) / distances[j]
                candidate = (distances[j] - dist) / distances[j]
                if abs(candidate) < reward:
                    reward = candidate
            self.last_distances[j] = dist

            i += 3
            j += 1

        # if abs(reward) < 0.3:
            # return reward

        # if reward > 0:
        #     if reward > 0.1:
        #         return -reward/1000
        
        if reward == 0.3:
            return 0
        return -reward/500

    def calc_dist_diff(self, obj1_position, obj2_position):
        """
        Calculate change in the distance between 2 objects in previous and in current step. Normalize the change by the value of distance in previous step.

        Params:
            :param obj1_position: (list) Position of the first object
            :param obj2_position: (list) Position of the second object
        Returns:
            :return norm_diff: (float) Normalized difference of distances between 2 objects in previsous and in current step
        """
        if self.prev_obj1_position is None and self.prev_obj2_position is None:
            self.prev_obj1_position = obj1_position
            self.prev_obj2_position = obj2_position
        self.prev_diff = self.task.calc_distance(self.prev_obj1_position, self.prev_obj2_position)

        current_diff = self.task.calc_distance(obj1_position, obj2_position)
        norm_diff = (self.prev_diff - current_diff) / self.prev_diff

        self.prev_obj1_position = obj1_position
        self.prev_obj2_position = obj2_position

        return norm_diff


class SparseReward(Reward):
    """
    Reward class for sparse reward signal

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(SparseReward, self).__init__(env, task)

    def reset(self):
        pass

    def compute(self, observation=None):
        """
        Compute sparse reward signal. Reward is 0 when goal is reached, -1 in every other step.

        Params:
            :param observation: Ignored
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        reward = -1

        if self.task.check_distance_threshold(observation):
            reward += 1.0

        self.rewards_history.append(reward)
        return reward
