from sumo.sumo_controller import *


class SumoAgent:
    def __init__(self, name: str, require_gui: bool, gamma=0.8, delta_t=5):
        self.sumo_controller = SumoController(name)
        self.sumo_cmd = ["sumo-gui" if require_gui else "sumo", "-c", self.sumo_controller.sumocfg]
        self.gamma = gamma
        self.delta_t = delta_t

        self._current_state = None
        self._cumulative_reward = torch.Tensor([0])
        self._current_reward = None
        self.current_action = None

    def take_action(self, action: list) -> None:
        """
        :param change_phase_or_not: A boolean list (length = num_junctions) indicating whether each junction change
                                    its phase or not.
                                    (False/0 represents stay put, True/1 represents change to next phase)
        :return: None
        """
        if action is None:
            raise TypeError("Action is None")
        if len(action) != self.sumo_controller.num_junctions:
            raise AttributeError("The length of input is not equal to num_junctions")
        self.current_action = action
        for i, junction_id in enumerate(self.sumo_controller.junctions_id):
            if self.current_action[i]:
                self.sumo_controller.set_tls_to_next_phase_at_junction(junction_id)

    def _calc_current_state(self):
        list_queue_length, list_updated_waiting_time, list_phase, list_vehicle_number = [], [], [], []
        for junction_id in self.sumo_controller.junctions_id:
            list_queue_length.append(
                self.sumo_controller.get_queue_length_at_junction(junction_id)
            )
            list_updated_waiting_time.append(
                self.sumo_controller.get_updated_waiting_time_at_junction(junction_id)
            )
            list_phase.append(
                self.sumo_controller.get_tls_current_phase_at_junction(junction_id)
            )
            list_vehicle_number.append(
                self.sumo_controller.get_vehicle_number_at_junction(junction_id)
            )

        tensor_queue_length = torch.Tensor(list_queue_length)
        tensor_updated_waiting_time = torch.Tensor(list_updated_waiting_time)
        tensor_phase = torch.Tensor(list_phase)
        tensor_vehicle_number = torch.Tensor(list_vehicle_number)

        tensor_state = torch.stack([tensor_queue_length,
                                    tensor_updated_waiting_time,
                                    tensor_phase,
                                    tensor_vehicle_number
                                    ]).T
        self._current_state = tensor_state

    def get_current_state(self) -> torch.Tensor:
        """
        :return: A 2D torch.Tensor in the form of [num_junctions *
                (queue_length, updated_waiting_time, phase, vehicle_number)]
        """
        self._calc_current_state()
        return self._current_state

    def _calc_current_reward(self):
        if self.current_action is None:
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
            current_reward[3] += c if self.current_action[i] != 0 else 0

            # 5.? Total number of vehicles $N$ that passed the intersection during time interval $\Delta t$
            # after the last action $a$
            # 6.? Total travel time of vehicles T that passed the intersection during time interval $\Delta t$
            # after the last action $a$
            vehicle_number, travel_time = self.sumo_controller.get_passed_vehicle_number_and_travel_time_at_junction(junction_id)
            current_reward[4] += vehicle_number
            current_reward[5] += travel_time

        self.current_reward = torch.sum(weight * current_reward).requires_grad_(False)
        self.current_action = None
        return

    def get_current_reward(self):
        """
        Get the reward for the last action
        :return: A torch.Tensor with only 1 element
        """
        return self.current_reward

    def _calc_cumulative_reward(self):
        self._cumulative_reward = self.gamma * self._cumulative_reward + self.current_reward

    def get_cumulative_reward(self) -> torch.Tensor:
        """
        Get the reward for the whole episode
        :return: A torch.Tensor with only 1 element
        """
        return self._cumulative_reward

    def all_travels_completed(self):
        return True if self.sumo_controller.get_vehicle_number_in_the_network() == 0 else False

    def sumo_start(self):
        traci.start(self.sumo_cmd)

    def sumo_end(self):
        traci.close()

    def step(self, action):

        # 1. Take action
        self.take_action(action)

        # 2. Perform one-step simulation and update vehicle
        for _ in range(self.delta_t):
            traci.simulationStep()
            self.sumo_controller.update_vehicle_enter_time()

        # 3. Calculate reward
        self._calc_current_reward()
        self._calc_cumulative_reward()
