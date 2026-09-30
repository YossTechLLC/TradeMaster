# [TradeMaster] thread caps must be set before numpy/torch are imported to take effect (upstream set
# them after 'import torch'). TM_THREADS (default 1, the upstream intent) sets all of them; the BOX runs
# many 1-thread processes side by side instead of one multi-threaded one.
import os
_THREADS = os.environ.get("TM_THREADS", "1")
for _v in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = _THREADS

import pathlib
import copy
import sys
import random
import argparse

import torch
torch.set_num_threads(int(_THREADS))  # [TradeMaster]
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # [TradeMaster] was the typo F_ENABLE_ONEDNN_OPTS

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.7)
parser.add_argument("--epsilon_end",type=float,default=0.3)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=0.2 / 1000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--alpha",type=float,default=0.5)
parser.add_argument("--beta",type=int,default=5)
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--num_step",type=int,default=10)
# [TradeMaster] sub-agent checkpoints were hard-coded to result/low_level/ETHUSDT/best_model
# [TradeMaster] bar-count constants (upstream: 1-minute bars), for coarser bars
parser.add_argument("--context_window",type=int,default=360,
                    help="window of the slope_<w>/vol_<w> context columns (decomposition.py MACRO_CONTEXT_WINDOW)")
parser.add_argument("--memory_capacity",type=int,default=4320,
                    help="episodic memory size in steps (also when re-encoding starts)")
parser.add_argument("--subagent_path",type=str,default=None,
                    help="dir with {slope,vol}/{1,2,3}/best_model.pkl (default: result/low_level/<dataset>/best_model)")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.dataset=args.dataset
        self.num_step = args.num_step
        if "BTC" in self.dataset:
            self.max_holding_number=0.01
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")
        self.epoch_number = args.epoch_number
        
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['slope_{}'.format(args.context_window), 'vol_{}'.format(args.context_window)]  # [TradeMaster] was *_360
        # [TradeMaster] standardise the context features with train-split statistics: slope_360 is a raw price
        # slope (scale of the price) and vol_360 ~1e-3, both fed unnormalised into hyperagent.fc2
        clf_train = pd.read_feather(os.path.join(self.train_data_path, "train.feather"), columns=self.clf_list)
        self.clf_mean = clf_train.mean().to_dict()
        self.clf_std = {k: (v if v > 0 else 1.0) for k, v in clf_train.std().to_dict().items()}
        with open(os.path.join(self.model_path, "clf_normalisation.yaml"), "w") as f:
            yaml.safe_dump({"mean": self.clf_mean, "std": self.clf_std}, f)

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.slope_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)        
        # [TradeMaster] sub-agents come from --subagent_path (default: this dataset's best_model dir)
        subagent_path = args.subagent_path or os.path.join("./result/low_level", args.dataset, "best_model")
        model_list_slope = [os.path.join(subagent_path, "slope", str(i), "best_model.pkl") for i in (1, 2, 3)]
        model_list_vol = [os.path.join(subagent_path, "vol", str(i), "best_model.pkl") for i in (1, 2, 3)]
        missing = [p for p in model_list_slope + model_list_vol if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("sub-agent checkpoints missing (train them with ./run.sh train-low, "
                                    "or pass --subagent_path): {}".format(missing))
        self.slope_1.load_state_dict(
            torch.load(model_list_slope[0], map_location=self.device))
        self.slope_2.load_state_dict(
            torch.load(model_list_slope[1], map_location=self.device))
        self.slope_3.load_state_dict(
            torch.load(model_list_slope[2], map_location=self.device))
        self.vol_1.load_state_dict(
            torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(
            torch.load(model_list_vol[1], map_location=self.device))
        self.vol_3.load_state_dict(
            torch.load(model_list_vol[2], map_location=self.device))
        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()
        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()
        # [TradeMaster] the sub-agents are frozen (only the hyper-agent is optimised): don't track their grads
        for m in (self.slope_1, self.slope_2, self.slope_3, self.vol_1, self.vol_2, self.vol_3):
            m.requires_grad_(False)
        self.slope_agents = {
            0: self.slope_1,
            1: self.slope_2,
            2: self.slope_3
        }
        self.vol_agents = {
            0: self.vol_1,
            1: self.vol_2,
            2: self.vol_3
        }
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(),
                                          lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = args.epsilon_start
        self.memory_capacity = args.memory_capacity  # [TradeMaster] was 4320
        self.memory = episodicmemory(self.memory_capacity, 5, self.n_state_1, self.n_state_2, 64, self.device)
        self.args = args  # [TradeMaster] methods used the module-global `args` (only worked as __main__)
        self.alpha, self.beta = args.alpha, args.beta
        self._q_cache = None  # [TradeMaster] see _q_values

    def read_split(self, path):
        # [TradeMaster] read a whole-split file with the context features standardised (see __init__)
        df = pd.read_feather(path)
        for c in self.clf_list:
            df[c] = (df[c] - self.clf_mean[c]) / self.clf_std[c]
        return df

    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        
        return combined_q


    def update(self, replay_buffer):
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        with torch.no_grad():  # [TradeMaster] target side: no graph needed (argmax / target net only)
            w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
            w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])


        qs_current = [
                    self.slope_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[2](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[2](batch['state'], batch['state_trend'], batch['previous_action'])
        ]
        with torch.no_grad():  # [TradeMaster]
            qs_next = [
                        self.slope_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                        self.slope_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                        self.slope_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                        self.vol_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                        self.vol_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                        self.vol_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
            ]
        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        with torch.no_grad():  # [TradeMaster]
            a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
            q_nexts = self.calculate_q(w_next, qs_next)
            q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        loss = td_error + self.alpha * memory_error + self.beta * KL_loss  # [TradeMaster] was args.*
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_counter += 1
        return td_error.cpu(), memory_error.cpu(), KL_loss.cpu(), torch.mean(q_current.cpu()), torch.mean(q_target.cpu())

    def _q_values(self, state, state_trend, state_clf, info):
        # [TradeMaster] one no-grad pass over the six sub-agents + hyper-agent (act/q_estimate built autograd
        # graphs before). q_estimate(s_) and the next step's act(s_) get the very same inputs, so the result
        # is reused as long as no update() has changed the weights in between.
        c = self._q_cache
        if (c is not None and c[0] is state and c[1] is state_trend and c[2] is state_clf
                and c[3] == info["previous_action"] and c[4] == self.update_counter):
            return c[5]
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
        self._q_cache = (state, state_trend, state_clf, info["previous_action"], self.update_counter, actions_value)
        return actions_value

    def act(self, state, state_trend, state_clf, info):
        if np.random.uniform() < (1-self.epsilon):
            actions_value = self._q_values(state, state_trend, state_clf, info)  # [TradeMaster]
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action

    def q_estimate(self, state, state_trend, state_clf, info):
        actions_value = self._q_values(state, state_trend, state_clf, info)  # [TradeMaster]
        q = torch.max(actions_value, 1)[0].detach().cpu().numpy()
        
        return q

    def calculate_hidden(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        with torch.no_grad():
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()
        return hs


    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        self.replay_buffer = ReplayBuffer_High(self.args, self.n_state_1, self.n_state_2, self.n_action)  # [TradeMaster]
        # [TradeMaster] read once (was re-read every epoch); the env never modifies it
        train_df = self.read_split(os.path.join(self.train_data_path, "train.feather"))
        for sample in range(self.epoch_number):
            print('epoch ', epoch_counter + 1)
            self.df = train_df
            
            
            train_env = Training_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    clf_list=self.clf_list,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=random.choices(range(self.n_action), k=1)[0],
                    alpha = 0)
            s, s2, s3, info = train_env.reset()
            episode_reward_sum = 0
            
            while True:
                a = self.act(s, s2, s3, info)
                s_, s2_, s3_, r, done, info_ = train_env.step(a)
                hs = self.calculate_hidden(s, s2, info)
                q = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, info_)
                q_memory = self.memory.query(hs, a)
                if np.isnan(q_memory):
                    q_memory = q
                self.replay_buffer.store_transition(s, s2, s3, info['previous_action'], info['q_value'], a, r, s_, s2_, s3_, info_['previous_action'],
                                info_['q_value'], done, q_memory)
                self.memory.add(hs, a, q, s, s2, info['previous_action'])
                episode_reward_sum += r

                s, s2, s3, info = s_, s2_, s3_, info_
                step_counter += 1
                if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step):
                    for i in range(self.update_times):
                        td_error, memory_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                        if self.update_counter % self.q_value_memorize_freq == 1:
                            self.writer.add_scalar(
                                tag="td_error",
                                scalar_value=td_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="memory_error",
                                scalar_value=memory_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="KL_loss",
                                scalar_value=KL_loss,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="q_eval",
                                scalar_value=q_eval,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="q_target",
                                scalar_value=q_target,
                                global_step=self.update_counter,
                                walltime=None)
                    if step_counter > self.memory_capacity:  # [TradeMaster] was 4320
                        self.memory.re_encode(self.hyperagent)
                if done:
                    break
            episode_counter += 1
            final_balance, required_money = train_env.final_balance, train_env.required_money
            self.writer.add_scalar(tag="return_rate_train",
                                scalar_value=final_balance / (required_money),
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="final_balance_train",
                                scalar_value=final_balance,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="required_money_train",
                                scalar_value=required_money,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="reward_sum_train",
                                scalar_value=episode_reward_sum,
                                global_step=episode_counter,
                                walltime=None)
            epoch_return_rate_train_list.append(final_balance / (required_money))
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)
                

            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
            self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_counter,
                    walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_final_balance_train",
                scalar_value=mean_final_balance_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_required_money_train",
                scalar_value=mean_required_money_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_reward_sum_train",
                scalar_value=mean_reward_sum_train,
                global_step=epoch_counter,
                walltime=None,
                )
            epoch_path = os.path.join(self.model_path,
                                        "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            torch.save(self.hyperagent.state_dict(),
                        os.path.join(epoch_path, "trained_model.pkl"))  
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                    os.makedirs(val_path)
            return_rate_eval = self.val_cluster(epoch_path, val_path)
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = copy.deepcopy(self.hyperagent.state_dict())  # [TradeMaster] snapshot, not a live reference
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
        if best_model is None:  # [TradeMaster] e.g. --epoch_number 0; never torch.save(None)
            print("warning: no epoch was selected on validation; testing the current weights")
            best_model = copy.deepcopy(self.hyperagent.state_dict())
        # [TradeMaster] per exp/seed (was result/high_level/<dataset>/, shared by every run)
        best_model_path = os.path.join(self.model_path, 'best_model.pkl')
        torch.save(best_model, best_model_path)  # [TradeMaster] best_model is already a state_dict
        final_result_path = os.path.join(self.model_path, 'test')
        os.makedirs(final_result_path, exist_ok=True)
        self.test_cluster(best_model_path, final_result_path)


    def val_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = self.read_split(
            os.path.join(self.val_data_path, "val.feather"))  # [TradeMaster]
        
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
            slient=True)
        final_balance = val_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"),
                commission_fee_list)
        return_rate = final_balance / required_money
        # [TradeMaster] an agent that never buys has required_money == 0 -> nan/inf; score it as 0
        # (nan never beat the best score, so best_model stayed None and the final test crashed)
        if not np.isfinite(return_rate):
            return_rate = 0.0
        return return_rate

    def test_cluster(self, epoch_path, save_path):
        # [TradeMaster] train() passes the best_model.pkl file itself, not an epoch directory
        model_file = epoch_path if epoch_path.endswith(".pkl") else os.path.join(epoch_path, "trained_model.pkl")
        self.hyperagent.load_state_dict(
            torch.load(model_file))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = self.read_split(
            os.path.join(self.test_data_path, "test.feather"))  # [TradeMaster]
        
        test_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = test_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(
            slient=True)
        final_balance = test_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action.npy"), action_list)
        np.save(os.path.join(save_path, "reward.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history.npy"),
                commission_fee_list)
            


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
