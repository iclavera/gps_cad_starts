import imp
import numpy as np
import os; osp = os.path
import pdb
import random
import tensorflow as tf

from gps.agent.ros.agent_ros import AgentROS
from gps.sample.sample_list import SampleList
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy.theano_policy import ThPolicy


def get_var_values(sess):
    variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    values = sess.run(variables)
    return {str(variables[i].name): values[i] for i in range(len(variables))}

def take_policy_samples(agent, policy, conditions, n):
    return [SampleList([agent.sample(policy, cond, save=False, noisy=False) for _ in range(n)]) for cond in conditions]

def save_data(output_dir, sample_lists):
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)
    DataLogger().pickle(osp.join(output_dir, 'pol_sample_itr_test.pkl'), sample_lists)

def main(n):
    hyperparams = imp.load_source('hyperparams', 'hyperparams.py')
    # arch = imp.load_source('arch', 'arch.py')

    T = hyperparams.T
    dO, dU = 32 + 9*T + 1, 7
    output_dir = hyperparams.agent['policy_path']

    try:
        policy = ThPolicy(output_dir, hyperparams.agent['recurrent'])
    except:
        print 'Failed to restore model. Exiting.'
        exit()

    # restored_values = get_var_values(policy_opt.)

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    agent = AgentROS(hyperparams.config['agent'])
    conditions = range(30)
    sample_lists = take_policy_samples(agent, policy, conditions, n)
    save_data(output_dir, sample_lists)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test a trained policy')
    # parser.add_argument('attention', metavar='ATTENTION')
    # parser.add_argument('structure', metavar='STRUCTURE')
    # parser.add_argument('-n', metavar='N', type=int, default=1)
    # args = parser.parse_args()
    main(1)
