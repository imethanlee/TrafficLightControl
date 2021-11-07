import torch

from sumo.sumo_controller import *


class SumoAgent:
    def __init__(self, name: str, require_gui: bool, gamma=0.8, delta_t=5):
        self._name = name
        self._require_gui = require_gui
        self.gamma = gamma
        self.delta_t = delta_t
        self._yellow_duration = 3
        if self._yellow_duration > self.delta_t:
            raise ValueError("Duration of Yellow Light MUST < delta_t")

        self.sumo_controller = SumoController(self._name)
        self.sumo_cmd = ["sumo-gui" if self._require_gui else "sumo", "-c", self.sumo_controller.sumocfg]
        self._current_state = torch.zeros([self.sumo_controller.num_junctions * 5])
        self._current_phase = torch.zeros([self.sumo_controller.num_junctions])
        self._next_phase = torch.zeros([self.sumo_controller.num_junctions])
        self._cumulative_reward = torch.Tensor([0])
        self._current_reward = None
        self._current_action = None

        self._yellow_cnt = -torch.ones([self.sumo_controller.num_junctions])
        self._step = 0

        self._metric_reward, self._list_metric_reward = 0, []
        self._metric_queue_length, self._list_metric_queue_length = torch.zeros([self.sumo_controller.num_junctions]), []
        self._metric_delay, self._list_metric_delay = torch.zeros([self.sumo_controller.num_junctions]), []

    def sumo_start(self):
        traci.start(self.sumo_cmd)

    def sumo_end(self):
        self.sumo_controller = SumoController(self._name)
        self.sumo_cmd = ["sumo-gui" if self._require_gui else "sumo", "-c", self.sumo_controller.sumocfg]

        self._current_state = torch.zeros([self.sumo_controller.num_junctions * 5])
        self._current_phase = torch.zeros([self.sumo_controller.num_junctions])
        self._next_phase = torch.zeros([self.sumo_controller.num_junctions])
        self._cumulative_reward = torch.Tensor([0])
        self._current_reward = None
        self._current_action = None

        self._yellow_cnt = -torch.ones([self.sumo_controller.num_junctions])
        self._step = 0

        self._metric_reward, self._list_metric_reward = 0, []
        self._metric_queue_length, self._list_metric_queue_length = torch.zeros([self.sumo_controller.num_junctions]), []
        self._metric_delay, self._list_metric_delay = torch.zeros([self.sumo_controller.num_junctions]), []

        traci.close()

    def get_dict_phase_to_int(self):
        return self.sumo_controller.get_dict_phase_to_int()

    def _yellow_control(self):
        for i in range(len(self._yellow_cnt)):
            if self._yellow_cnt[i] == self._yellow_duration:
                self._yellow_cnt[i] = -1
                junction_id = self.sumo_controller.junctions_id[i]
                self.sumo_controller.set_tls_to_next_phase_at_junction(junction_id)
        self._yellow_cnt[self._yellow_cnt >= 0] += 1

    def _take_action(self, action: int):
        """
        :param action: A number ranging from 0 to 2^{num_junctions} - 1
        :return: None
        """
        if action is None:
            raise TypeError("Action is None")
        action = self._int_to_bin(action)
        if len(action) != self.sumo_controller.num_junctions:
            raise AttributeError("The length of input is not equal to num_junctions")
        self._current_action = action
        for i, junction_id in enumerate(self.sumo_controller.junctions_id):
            if self._current_action[i]:
                self.sumo_controller.set_tls_to_next_phase_at_junction(junction_id)
                self._yellow_cnt[i] = 0

    def _int_to_bin(self, val):
        return [val >> d & 1 for d in range(self.get_num_junctions())][::-1]

    def _calc_current_state(self):
        list_queue_length, list_updated_waiting_time, list_current_phase, list_next_phase, list_vehicle_number \
            = [], [], [], [], []
        for junction_id in self.sumo_controller.junctions_id:
            list_queue_length.append(
                self.sumo_controller.get_queue_length_at_junction(junction_id)
            )
            list_updated_waiting_time.append(
                self.sumo_controller.get_updated_waiting_time_at_junction(junction_id)
            )
            list_vehicle_number.append(
                self.sumo_controller.get_vehicle_number_at_junction(junction_id)
            )
            list_current_phase.append(
                # self.sumo_controller.get_tls_current_phase_at_junction(junction_id)
                self.sumo_controller.get_tls_current_phase_at_junction(junction_id) / 2
            )
            list_next_phase.append(
                # self.sumo_controller.get_tls_next_phase_at_junction(junction_id)
                self.sumo_controller.get_tls_next_phase_at_junction(junction_id) / 2
            )

        tensor_queue_length = torch.Tensor(list_queue_length)
        tensor_updated_waiting_time = torch.Tensor(list_updated_waiting_time)
        tensor_vehicle_number = torch.Tensor(list_vehicle_number)
        tensor_current_phase = torch.Tensor(list_current_phase)
        tensor_next_phase = torch.Tensor(list_next_phase)

        tensor_state = torch.cat([tensor_queue_length,
                                  tensor_updated_waiting_time,
                                  tensor_vehicle_number,
                                  tensor_current_phase,
                                  tensor_next_phase,
                                  ], dim=0)

        self._current_phase = tensor_current_phase
        self._next_phase = tensor_next_phase
        self._current_state = tensor_state

    def get_current_state(self):
        """
        IMPORTANT: This method do not include any calculation
        :return: A 2D torch.Tensor in the form of [num_junctions *
                (queue_length, updated_waiting_time, vehicle_number, current_phase, next_phase)]
        """
        return self._current_state

    def get_current_phase(self):
        return self._current_phase

    def get_next_phase(self):
        return self._next_phase

    def _calc_current_reward(self):
        if self._current_action is None:
            raise RuntimeError("Take action first and then get reward!")

        current_reward = torch.zeros([6])
        weight = torch.Tensor([-0.25, -0.25, -0.25, -5.00, 1.00, 1.00])

        for i, junction_id in enumerate(self.sumo_controller.junctions_id):
            # 1. Sum of queue length $L$ over all approaching lanes
            current_reward[0] += self.sumo_controller.get_queue_length_at_junction(junction_id)

            # 2. Sum of delay $D$ over all approaching lanes
            current_reward[1] += self.sumo_controller.get_delay_at_junction(junction_id)

            # 3. Sum of updated waiting time W over all approaching lanes
            current_reward[2] += self.sumo_controller.get_updated_waiting_time_at_junction(junction_id)

            # 4. Indicator of light switches $C$
            c = 1
            current_reward[3] += c if self._current_action[i] != 0 else 0

            # 5. Total number of vehicles $N$ that passed the intersection during time interval $\Delta t$
            # after the last action $a$
            # 6. Total travel time of vehicles T that passed the intersection during time interval $\Delta t$
            # after the last action $a$
            vehicle_number, travel_time = self.sumo_controller.get_passed_vehicle_number_and_travel_time_at_junction(
                junction_id)
            current_reward[4] += vehicle_number
            current_reward[5] += travel_time

        self._current_reward = torch.sum(weight * current_reward)

        self._metric_reward += self._current_reward
        self._list_metric_reward.append(self._metric_reward)
        self._metric_queue_length += current_reward[0]
        self._list_metric_queue_length.append(self._metric_queue_length.numpy())
        self._metric_delay += current_reward[1]
        self._list_metric_delay.append(self._metric_delay.numpy())

        self.current_action = None

    def get_current_reward(self):
        """
        Get the reward for the last action
        :return: A torch.Tensor with only 1 element
        """
        return self._current_reward

    def _calc_cumulative_reward(self):
        """
        Warning: to be modified !!!
        :return:
        """
        self._cumulative_reward = self.gamma * self._cumulative_reward + self._current_reward

    def get_cumulative_reward(self):
        """
        Get the reward for the whole episode
        :return: A torch.Tensor with only 1 element
        """
        return self._cumulative_reward

    def metric_avg_reward(self):
        avg_reward = self._metric_reward / self._step
        avg_reward_over_time = torch.Tensor(self._list_metric_reward)
        # t = torch.Tensor([1 / i for i in range(self._step // self.delta_t)])
        # avg_reward_over_time = (t * avg_reward_over_time.T).T
        return avg_reward, avg_reward_over_time

    def metric_avg_queue_length(self):
        avg_queue_length = self._metric_queue_length / self._step
        avg_queue_length_over_time = torch.Tensor(self._list_metric_queue_length)
        # t = torch.Tensor([1 / i for i in range(self._step // self.delta_t)])
        # avg_queue_length_over_time = (t * avg_queue_length_over_time.T).T
        return avg_queue_length, avg_queue_length_over_time

    def metric_avg_delay(self):
        avg_delay = self._metric_delay / self._step
        avg_delay_over_time = torch.Tensor(self._list_metric_delay)
        # t = torch.Tensor([1 / i for i in range(self._step // self.delta_t)])
        # avg_delay_over_time = (t * avg_delay_over_time.T).T
        return avg_delay, avg_delay_over_time

    def get_current_time(self):
        return self.sumo_controller.get_current_time()

    def get_num_junctions(self):
        return self.sumo_controller.num_junctions

    def get_num_actions(self):
        return 2 ** self.sumo_controller.num_junctions

    def get_num_phases(self):
        return len(self.sumo_controller.get_dict_phase_to_int().keys())

    def get_state_dim(self):
        return self.sumo_controller.num_junctions * 5

    def all_travels_completed(self):
        return True if self.sumo_controller.get_vehicle_number_in_the_network() == 0 else False

    def step(self, action):
        # 1. Take action
        self._take_action(action)

        # 2. Perform delta_t-step simulation and update vehicle
        for _ in range(self.delta_t):
            self._yellow_control()
            traci.simulationStep()
            self.sumo_controller.update_vehicle_enter_time()
        self._step += self.delta_t

        # 3. Calculate state/reward
        self._calc_current_state()
        self._calc_current_reward()
        self._calc_cumulative_reward()
