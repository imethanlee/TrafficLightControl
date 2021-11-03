from sumo.sumo_agent import *


def fixed_policy_simulate(args):
    sumo_agent = SumoAgent(args.test_case_name,
                           require_gui=args.require_gui,
                           gamma=args.gamma,
                           delta_t=args.delta_t)
    sumo_agent.sumo_start()
    step = 0
    action = 2 ** sumo_agent.get_num_junctions() - 1
    while True:
        # TODO: Define some fixed policy
        current_state = sumo_agent.get_current_state()

        sumo_agent.step(action)

        next_state = sumo_agent.get_current_state()
        # current_reward = sumo_agent.get_current_reward()
        # cumulative_reward = sumo_agent.get_cumulative_reward()
        if sumo_agent.all_travels_completed():
            break
        step += sumo_agent.delta_t

    sumo_agent.sumo_end()
