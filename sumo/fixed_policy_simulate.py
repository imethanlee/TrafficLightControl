from sumo.sumo_agent import *


def fixed_policy_simulate(args):
    sumo_agent = SumoAgent(args)
    sumo_agent.sumo_start()
    step = 0
    while True:
        # TODO: Define some fixed policy
        # current_state = sumo_agent.get_current_state()

        if step % 20 == 0:
            action = 2 ** sumo_agent.get_num_junctions() - 1
        else:
            action = 0
        sumo_agent.step(action)

        # next_state = sumo_agent.get_current_state()
        # current_reward = sumo_agent.get_current_reward()
        # cumulative_reward = sumo_agent.get_cumulative_reward()
        if sumo_agent.all_travels_completed():
            break
        step += sumo_agent.delta_t
    reward, _ = sumo_agent.metric_avg_reward()
    print("reward: ", reward)
    q_length, _ = sumo_agent.metric_avg_queue_length()
    print("queue length: ", q_length)
    delay, _ = sumo_agent.metric_avg_delay()
    print("delay: ", delay)
    duration = sumo_agent.metric_avg_duration()
    print("duration: ", duration)
    print("step: ", step)
    sumo_agent.sumo_end()
