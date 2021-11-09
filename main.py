from sumo.sumo_agent import *
from sumo.fixed_policy_simulate import *
import argparse
from net_agent import NetAgent
import logging
import os
import sys
from os.path import join


def make_logger(name, save_dir, save_filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_time_stamp_str():
    """Return time stamp in string format"""
    import time
    import datetime
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    return st


def create_dir_with_st(path):
    """Create folder with name of timestamp, return folder path"""
    from os.path import join as opj
    from os import makedirs
    path = opj(path, get_time_stamp_str())
    makedirs(path, exist_ok=True)
    return path


# TODO: 参考下面这个method来 1.获取state 2.执行action 3.获取reward
def dqn_simulate(args, ckpt_path):
    # Specify the traffic environment
    sumo_agent = SumoAgent(args.test_case_name,
                           require_gui=args.require_gui,
                           gamma=args.gamma,
                           delta_t=args.delta_t)
    net_agent = NetAgent(args, sumo_agent)
    run_counts = args.run_counts
    all_time = run_counts
    avg_reward, avg_q_length, avg_delay = 0, float('inf'), float('inf')
    for epoch in range(args.max_epoch):
        is_val = False

        # Start interacting with the SUMO environment
        sumo_agent.sumo_start()
        step = 0
        net_agent.reset_update_count()

        current_time = sumo_agent.get_current_time()
        while current_time < run_counts:
            """#####################################################"""
            """####   1. Get the current state                  ####"""
            """#####################################################"""

            current_state = sumo_agent.get_current_state()
            current_phase = sumo_agent.get_current_phase()
            cur_phase = torch.tensor([sumo_agent.get_dict_phase_to_int()[tuple(current_phase.tolist())]])

            action = torch.tensor([net_agent.choose(current_time, current_state, cur_phase, is_val)])

            """#####################################################"""
            """####   2. Take your action                       ####"""
            """#####################################################"""
            """ (a boolean torch.Tensor/list with length num_junctions) """

            # get reward from sumo agent
            sumo_agent.step(action)

            """#####################################################"""
            """####  3. Get step/cumulative reward/next state   ####"""
            """#####################################################"""

            next_state = sumo_agent.get_current_state()
            next_phase = sumo_agent.get_next_phase()
            nex_phase = torch.tensor([sumo_agent.get_dict_phase_to_int()[tuple(next_phase.tolist())]])
            current_reward = torch.FloatTensor([sumo_agent.get_current_reward()])
            # cumulative_reward = sumo_agent.get_cumulative_reward()

            net_agent.remember(current_state, action, current_reward, next_state, cur_phase, nex_phase)

            loss = net_agent.trainer()
            logger.info('[INFO] Time {} Loss {} in epoch[{}/{}]'.format(current_time, loss, epoch, args.max_epoch))
            """#####################################################"""
            """########                END              ###########"""
            """#####################################################"""
            if sumo_agent.all_travels_completed():
                break
            step += sumo_agent.delta_t
            current_time = sumo_agent.get_current_time()
        sumo_agent.sumo_end()

        logger.info('\n[INFO] End of Epoch Validation...')

        sumo_agent.sumo_start()
        step = 0
        net_agent.reset_update_count()
        with torch.no_grad():
            is_val = True

            current_time = sumo_agent.get_current_time()
            while current_time < run_counts:

                current_state = sumo_agent.get_current_state()
                current_phase = sumo_agent.get_current_phase()
                cur_phase = torch.tensor([sumo_agent.get_dict_phase_to_int()[tuple(current_phase.tolist())]])

                action = torch.tensor([net_agent.choose(current_time, current_state, cur_phase, is_val)])

                # get reward from sumo agent
                sumo_agent.step(action)

                step += sumo_agent.delta_t
                current_time = sumo_agent.get_current_time()
                if sumo_agent.all_travels_completed():
                    # if sumo_agent.get_current_time() < all_time:
                    #     all_time = sumo_agent.get_current_time()
                    #     net_agent.save_model(join(ckpt_path, 'best_reward.pth'))
                    reward, _ = sumo_agent.metric_avg_reward()
                    if reward > avg_reward:
                        avg_reward = reward
                        net_agent.save_model(join(ckpt_path, 'best_reward.pth'))
                    q_length, _ = sumo_agent.metric_avg_queue_length()
                    if q_length[0] < avg_q_length:
                        avg_q_length = q_length[0]
                        net_agent.save_model(join(ckpt_path, 'best_q_len.pth'))
                    delay, _ = sumo_agent.metric_avg_delay()
                    if delay[0] < avg_delay:
                        avg_delay = delay[0]
                        net_agent.save_model(join(ckpt_path, 'best_delay.pth'))
                    break
            sumo_agent.sumo_end()
            logger.info('[INFO] reward {}, q_length {}, delay {} in epoch[{}/{}]'.format(reward, q_length, delay, epoch, args.max_epoch))
    dqn_test(sumo_agent, run_counts, net_agent, ckpt_path)
    # End interacting with the SUMO environment


def dqn_test(sumo_agent, run_counts, net_agent, ckpt_path):
    step = 0
    net_agent.load_model(join(ckpt_path, 'best_reward.pth'))
    with torch.no_grad():
        is_val = True

        current_time = sumo_agent.get_current_time()
        while current_time < run_counts:

            current_state = sumo_agent.get_current_state()
            current_phase = sumo_agent.get_current_phase()
            cur_phase = torch.tensor([sumo_agent.get_dict_phase_to_int()[tuple(current_phase.tolist())]])

            action = torch.tensor([net_agent.choose(current_time, current_state, cur_phase, is_val)])

            # get reward from sumo agent
            sumo_agent.step(action)

            step += sumo_agent.delta_t
            current_time = sumo_agent.get_current_time()
            if sumo_agent.all_travels_completed():
                # if sumo_agent.get_current_time() < all_time:
                #     all_time = sumo_agent.get_current_time()
                #     net_agent.save_model(join(ckpt_path, 'best_reward.pth'))
                reward, _ = sumo_agent.metric_avg_reward()
                q_length, _ = sumo_agent.metric_avg_queue_length()
                delay, _ = sumo_agent.metric_avg_delay()
                logger.info('[INFO] Test_reward {}, Test_q_length {}, Test_delay {}'.format(reward, q_length, delay))
                break
        sumo_agent.sumo_end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Test Case Name List
    # 1. "./data/self_defined_1/osm": small-scale self-defined network
    # 2. "./data/nus1/osm": NUS network 1
    # 3. "./data/nus2/osm": NUS network 2
    parser.add_argument('--require_gui', type=bool, default=False)
    parser.add_argument('--test_case_name', type=str, default="./data/self_defined_1/osm")
    parser.add_argument('--log_path', type=str, default="./log")
    # parser.add_argument('--num_phases', type=int, default=16)
    # parser.add_argument('--num_actions', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', default=50, type=int)

    parser.add_argument('--run_counts', default=72000, type=int)

    parser.add_argument('--delta_t', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # dqn_simulate()
    # fixed_policy_simulate(args)
    logger_path = create_dir_with_st(args.log_path)
    logger = make_logger('Logger file for action recognition', logger_path, 'log')

    ckpt_path = join(logger_path, 'checkpoints')

    os.makedirs(ckpt_path, exist_ok=True)

    torch.manual_seed(args.seed)
    dqn_simulate(args, ckpt_path)
    # fixed_policy_simulate(args)
    # np.random.seed(opt.seed)
