import torch
import numpy as np
from collections import deque, namedtuple
import random
import pdb


def get_ada(ada,decay_freq=2,ada_counter=0, decay_coffient=0.5):
    if ada_counter % decay_freq==1:
        ada = decay_coffient*ada
    return ada


def get_epsilon( epsilon,max_epsilon=1, epsilon_counter=0, decay_freq=2,decay_coffient=0.5):
    if epsilon_counter%decay_freq == 1:
        epsilon =epsilon+(max_epsilon-epsilon)*decay_coffient
    return epsilon

class LinearDecaySchedule(object):
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, t):
        return max(self.end_epsilon, self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (t / self.decay_length))


def feature_lists(dataset):
    # [TradeMaster] per-dataset input profile: data/<dataset>/feature_list/ (written by profiles/build.py),
    # falling back to the upstream lists in data/feature_list/
    import os
    d = os.path.join('./data', dataset, 'feature_list')
    if not os.path.exists(os.path.join(d, 'single_features.npy')):
        d = './data/feature_list'
    single = np.load(os.path.join(d, 'single_features.npy'), allow_pickle=True).tolist()
    trend = np.load(os.path.join(d, 'trend_features.npy'), allow_pickle=True).tolist()
    print("input features from {}: {} single, {} trend".format(d, len(single), len(trend)))
    return single, trend
