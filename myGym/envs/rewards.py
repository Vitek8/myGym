import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import results_plotter
import os


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
            results_plotter.EPISODES_WINDOW = 50
            results_plotter.plot_curves(
                [(np.arange(self.env.episode_steps), np.asarray(self.rewards_history[-self.env.episode_steps:]))],
                'step', 'Step rewards')
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
            results_plotter.EPISODES_WINDOW = 10
            results_plotter.plot_curves([(np.arange(self.env.episode_number),
                                          np.asarray(self.env.episode_final_reward[-self.env.episode_number:]))],
                                        'episode', 'Episode rewards')
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


class SwitchReward(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects
    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task):
        super(SwitchReward, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None

        self.x_obj = None
        self.y_obj = None
        self.z_obj = None
        self.x_bot = None
        self.y_bot = None
        self.z_bot = None

        self.x_obj_curr_pos = None
        self.y_obj_curr_pos = None
        self.z_obj_curr_pos = None
        self.x_bot_curr_pos = None
        self.y_bot_curr_pos = None
        self.z_bot_curr_pos = None

        self.height = None
        self.offset = None
        self.prev_angle = None

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3]) / 2)]
        o2 = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3]) / 2):-3]
        gripper_position = self.get_accurate_gripper_position(observation[3:6])
        self.set_variables(o1, gripper_position)
        self.set_offset(x=-0.1, z=0.25)

        if self.x_obj > 0:
            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], [-0.7, self.y_obj, self.z_obj],
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=1)
            w = self.calc_direction_3d(self.x_obj, self.y_obj, self.z_obj, -0.7, self.y_obj, self.z_obj, self.x_bot_curr_pos,
                                       self.y_bot_curr_pos, self.z_bot_curr_pos)

        else:
            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], [0.7, self.y_obj, self.z_obj],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1)

            w = self.calc_direction_3d(self.x_obj, self.y_obj, self.z_obj, -0.7, self.y_obj, self.z_obj, self.x_bot_curr_pos,
                                          self.y_bot_curr_pos, self.z_bot_curr_pos)

        d = self.abs_diff()
        a = self.calc_angle_reward(self.get_angle())

        k_w = 0.4
        k_d = 0.1
        k_a = 0.1

        reward = - k_w * w - k_d * d + a * k_a
        self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], gripper_position,
                                    lineColorRGB=(0.31, 0.78, 0.47), lineWidth=3, lifeTime=0.03)
        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        self.env.p.addUserDebugText(f"reward-{reward:.5f}, w-{w * k_w}, d-{d * k_d}, a-{a * k_a}",
                                    [1, 1, 1], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])
        return reward

    def reset(self):
        """
        Reset stored value of distance be
        """
        self.prev_obj1_position = None
        self.prev_obj2_position = None

        self.x_obj = None
        self.y_obj = None
        self.z_obj = None
        self.x_bot = None
        self.y_bot = None
        self.z_bot = None

        self.x_obj_curr_pos = None
        self.y_obj_curr_pos = None
        self.z_obj_curr_pos = None
        self.x_bot_curr_pos = None
        self.y_bot_curr_pos = None
        self.z_bot_curr_pos = None

        self.height = None
        self.offset = None
        self.prev_angle = None

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction = [0, 0, 0.1]  # length is 0.1
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]],
                      [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]],
                      [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        orientation_vector = m.dot(direction)  # length is 0.1
        gripper_position = np.add(gripper_position, orientation_vector)
        return gripper_position

    def set_variables(self, o1, o2):
        if self.x_obj is None:
            self.x_obj = o1[0]

        if self.y_obj is None:
            self.y_obj = o1[1]

        if self.z_obj is None:
            self.z_obj = o1[2]

        if self.x_bot is None:
            self.x_bot = o2[0]

        if self.y_bot is None:
            self.y_bot = o2[1]

        if self.z_bot is None:
            self.z_bot = o2[2]

        self.x_obj_curr_pos = o1[0]
        self.y_obj_curr_pos = o1[1]
        self.z_obj_curr_pos = o1[2]
        self.x_bot_curr_pos = o2[0]
        self.y_bot_curr_pos = o2[1]
        self.z_bot_curr_pos = o2[2]
        self.height = sqrt(o1[2] ** 2 + o2[2] ** 2)

    def set_offset(self, x=0.0, y=0.0, z=0.0):
        if self.offset is None:
            self.offset = True
            if self.x_obj > 0:
                self.x_obj -= x
                self.y_obj += y
                self.z_obj += z
            else:
                self.x_obj += x
                self.y_obj += y
                self.z_obj += z

    def calc_direction_2d(self, x1, y1, x2, y2, x3, y3):
        x = x1 + ((x1 - x2) * (x1 * x2 + x1 * x3 - x2 * x3 + y1 * y2 + y1 * y3 - y2 * y3 - x1 ** 2 - y1 ** 2)) / (
                    x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2)
        y = y1 + ((y1 - y2) * (x1 * x2 + x1 * x3 - x2 * x3 + y1 * y2 + y1 * y3 - y2 * y3 - x1 ** 2 - y1 ** 2)) / (
                    x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2)
        d = sqrt((x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2) ** 2 / (
                    x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2))
        return d

    def calc_direction_3d(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        x = x1 - ((x1 - x2) * (
                    x1 * (x1 - x2) - x3 * (x1 - x2) + y1 * (y1 - y2) - y3 * (y1 - y2) + z1 * (z1 - z2) - z3 * (
                        z1 - z2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        y = y1 - ((y1 - y2) * (
                    x1 * (x1 - x2) - x3 * (x1 - x2) + y1 * (y1 - y2) - y3 * (y1 - y2) + z1 * (z1 - z2) - z3 * (
                        z1 - z2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        z = z1 - ((z1 - z2) * (
                    x1 * (x1 - x2) - x3 * (x1 - x2) + y1 * (y1 - y2) - y3 * (y1 - y2) + z1 * (z1 - z2) - z3 * (
                        z1 - z2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        d = sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2)

        return d

    def abs_diff(self):
        x_diff = self.x_obj_curr_pos - self.x_bot_curr_pos
        y_diff = self.y_obj_curr_pos - self.y_bot_curr_pos
        z_diff = self.z_obj_curr_pos - self.z_bot_curr_pos

        abs_diff = sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        return abs_diff

    def is_touched(self, contact_points):
        ret = None
        for val in contact_points:

            if val != ():
                ret = True

        return ret

    def calc_dist(self, o1, o2, threshold):
        x = o1[0] - o2[0]
        y = o1[1] - o2[1]
        z = o1[2] - o2[2]

        print(x, y)
        if abs(x) < threshold and abs(y) < threshold:
            return 1 / math.sqrt(x ** 2 + y ** 2) * 10
        else:
            return 1 / math.sqrt(x ** 2 + y ** 2) - 1

    def get_angle(self):
        if self.task.task_type == "switch":
            if len(self.task.current_task_objects) != 2:
                raise ("not expected number of objects")

            o1 = self.task.current_task_objects[0]
            o2 = self.task.current_task_objects[1]

            if o1 == self.env.robot:
                robot = o1
                switch = o2
            else:
                robot = o2
                switch = o1

            p = self.env.p
            pos = p.getJointState(switch.get_uid(), 0)
            angle = int(pos[0] * 180 / math.pi)  # in degrees
            return abs(angle)

    def calc_angle_reward(self, angle):
        if self.task.task_type == "switch":
            if self.prev_angle is None:
                self.prev_angle = angle

            if angle < 0:
                k = angle // 2

            else:
                k = angle // 2

            reward = k * angle

            if reward >= 162:
                reward += 50
            reward /= 100
            if self.prev_angle == angle:
                reward = 0

            # if angle != 0 and angle != 1:
            #     if angle > self.prev_angle:
            #         pass
            #         # print("úhel se zvětšuje")
            #
            #     if angle < self.prev_angle:
            #         reward = -reward
            #         # print("úhel se zmenšuje")

            self.prev_angle = angle

            return reward
        else:
            raise ("not expected to use this function")


class SwitchReward_old(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects
    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task):
        super(SwitchReward_old, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None
        self.prev_angle = None

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3]) / 2)]
        o2 = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3]) / 2):-3]
        height = math.sqrt(o1[2] ** 2 + o2[2] ** 2)
        angle = self.get_angle()
        self.task.check_distance_threshold(observation=observation)
        if height >= 0.3 and height <= 0.5:
            reward = self.calc_dist(o1, o2, 0.2) + self.calc_angle_reward(angle)
        else:
            reward = self.calc_dist(o1, o2, 0.2) - 1

        self.rewards_history.append(reward)
        return reward

    def calc_dist(self, o1, o2, threshold):
        x = o1[0] - o2[0]
        y = o1[1] - o2[1]
        # print(x, y)
        if abs(x) < threshold and abs(y) <= threshold:
            return 1 / math.sqrt(x ** 2 + y ** 2) * 10
        else:
            return 1 / math.sqrt(x ** 2 + y ** 2)

    def get_angle(self):
        if self.task.task_type == "switch":
            if len(self.task.current_task_objects) != 2:
                raise ("not expected number of objects")

            o1 = self.task.current_task_objects[0]
            o2 = self.task.current_task_objects[1]
            if o1 == self.env.robot:
                robot = o1
                switch = o2
            else:
                robot = o2
                switch = o1
            p = self.env.p
            pos = p.getJointState(switch.get_uid(), 0)
            angle = int(pos[0] * 180 / math.pi)  # in degrees
            return abs(angle)

    def calc_angle_reward(self, angle):
        if self.task.task_type == "switch":
            if self.prev_angle is None:
                self.prev_angle = angle

            if angle < 0:
                k = angle // 2

            else:
                k = angle // 2

            reward = k * angle

            if reward >= 162:
                reward += 50

            if self.prev_angle == angle:
                reward = 0

            self.prev_angle = angle
            return reward
        else:
            raise ("not expected to use this function")


class SwitchReward_not_working(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task):
        super(SwitchReward_not_working, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None
        self.prev_reward = None
        self.reward = None
        self.angle = None
        self.prev_angle = None

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3]) / 2)]
        o2 = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3]) / 2):-3]
        diff_height = math.sqrt(o1[2] ** 2 + o2[2] ** 2)
        reward = 0
        # print(f"height = {diff_height:.5f}, distance = {diff_distance:.5f}, threshold = {self.calc_dist_diff(o1, o2):.5f}")

        if self.task.check_distance_threshold(observation):
            reward += 100
        else:
            reward -= 100

        if diff_height >= 0.2 and diff_height <= 0.4:
            reward = 1 / self.calc_dist_diff(o1, o2)
        else:
            reward -= 100

        self.task.check_angle_threshold()
        # if not (self.check_reward_change()):
        #     self.task.check_angle_threshold()
        #     reward += self.reward

        # else:
        #     reward -= 100

        self.rewards_history.append(reward)
        print(f"reward: {reward},"
              f"height: {diff_height:.5f},"
              # f"check_angle: {self.check_reward_change()},"
              f"angle: {self.calc_angle_reward()}")
        return reward

    # def check_reward_change(self):
    #     val = False
    #     if self.reward is None and self.prev_reward is None:
    #         self.reward = self.calc_angle_reward()
    #         self.prev_reward = self.reward
    #     else:
    #         self.reward = self.calc_angle_reward()
    #
    #     if self.reward == self.prev_reward:
    #         val = True
    #
    #     self.prev_reward = self.reward
    #     return val

    def calc_angle_reward(self):
        if self.task.task_type == "switch":
            if len(self.task.current_task_objects) != 2:
                print("not expected number of objects")
            o1 = self.task.current_task_objects[0]
            o2 = self.task.current_task_objects[1]
            if o1 == self.env.robot:
                robot = o1
                switch = o2
            else:
                robot = o2
                switch = o1
            p = self.env.p
            pos = p.getJointState(switch.get_uid(), 0)
            print(pos)
            angle = int(pos[0] * 180 / math.pi)  # in degrees
            if self.prev_angle is None:
                self.prev_angle = angle

            if self.angle < 0:
                k = angle // 2

            else:
                k = angle // 2

            reward = k * angle

            if reward >= 162:
                reward += 50

            if self.prev_angle == angle:
                reward = 0

            self.prev_angle = angle
            return reward
        else:
            print("not expected to use this function")


class SwitchReward_2(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task):
        super(SwitchReward, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None
        self.prev_reward = None
        self.reward = None

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        # print(observation)
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3]) / 2)]
        o2 = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3]) / 2):-3]
        diff_height = math.sqrt(o1[2] ** 2 + o2[2] ** 2)
        diff_y = math.sqrt(o1[1] ** 2 + o2[1] ** 2)
        diff_x = math.sqrt(o1[0] ** 2 + o2[0] ** 2)

        reward = 0
        threshold = self.task.check_distance_threshold(observation)
        angle_reward = self.task.calc_angle_reward()

        if threshold:
            reward += 100
        else:
            reward -= 10

        if diff_height >= 0.2 and diff_height <= 0.4:
            reward += self.task.calc_distance(o1, o2)
        else:
            reward -= 10

        if angle_reward > 0:
            reward += angle_reward
        else:
            reward -= 10
        # print(f"threshold: {threshold},"
        #       f"height: {diff_height:.5f},"
        #       f"angle: {angle_reward}",
        #       f"reward: {reward}")
        print(f"[{diff_x:.3f},{diff_y:.3f},{diff_height:.3f}]")
        self.rewards_history.append(reward)
        return reward


class SwitchReward_fce(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task):
        super(SwitchReward, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None
        self.prev_reward = None
        self.reward = None

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3]) / 2)]
        o2 = observation[3:6] if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3]) / 2):-3]
        diff_height = math.sqrt(o1[2] ** 2 + o2[2] ** 2)
        diff_distance = math.sqrt(o1[0] ** 2 + o2[0] ** 2)
        # print(f"height = {diff_height:.5f}, distance = {diff_distance:.5f}, threshold = {self.calc_dist_diff(o1, o2):.5f}")

        if (self.task.check_distance_threshold(observation)):
            print("threshold reached")
            if diff_height >= 0.2 and diff_height <= 0.37:
                # print("height reached", diff_height)
                self.task.check_distance_threshold(observation)
                reward = 1 / self.task.calc_distance(o1, o2)

                if self.check_reward_change(self.reward, self.prev_reward):
                    reward += self.task.calc_angle_reward()

            else:
                # print("not reached", diff_height)
                self.task.check_distance_threshold(observation)
                reward = 0
        else:
            # print("not reached")
            reward = self.task.calc_angle_reward()
        # print("reward: ", reward)
        self.rewards_history.append(reward)
        print(reward)
        return reward

    def check_reward_change(self, reward, prev_reward):
        val = None
        self.reward = reward
        self.prev_reward = prev_reward
        if reward is None and prev_reward is None:
            self.reward = self.task.calc_angle_reward()
            self.prev_reward = self.task.calc_angle_reward()
        if self.reward == self.prev_reward:
            val = True

        self.prev_reward = self.reward
        return False if val is None else True


class ComplexDistanceReward(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 3 objects, e.g. 2 objects and gripper for complex tasks

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task):
        super(ComplexDistanceReward, self).__init__(env, task)
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

        norm_diff = (prev_diff_13 - current_diff_13) / prev_diff_13 + (
                prev_diff_23 - current_diff_23) / prev_diff_23 + 10 * (
                            prev_diff_12 - current_diff_12) / prev_diff_12

        self.prev_obj1_position = obj1_position
        self.prev_obj2_position = obj2_position
        self.prev_obj3_position = obj3_position

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
