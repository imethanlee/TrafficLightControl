from sumo.sumo_agent import *


def fixed_policy_simulate(args):
    sumo_agent = SumoAgent(args.test_case_name,
                           require_gui=args.require_gui,
                           gamma=args.gamma,
                           delta_t=args.delta_t)
    sumo_agent.sumo_start()
    step = 0
    action = torch.Tensor([1 for _ in range(sumo_agent.sumo_controller.num_junctions)])
    while True:
        # TODO: Define some fixed policy
        sumo_agent.step(action)
        # current_reward = sumo_agent.get_current_reward()
        # cumulative_reward = sumo_agent.get_cumulative_reward()

        if sumo_agent.all_travels_completed():
            break
        step += sumo_agent.delta_t

    sumo_agent.sumo_end()
