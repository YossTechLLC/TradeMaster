import pandas as pd
import numpy as np
import os
import re
import torch
import argparse

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--root_path",
    type=str,
    default="result_risk/BTCUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="result_risk/BTCUSDT/potential_model",
    help="the number of transcation we store in one memory",
)
# [TradeMaster] was hard-coded seed_12345
parser.add_argument("--seed", type=int, default=12345, help="the low-level training seed to pick from")


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

#TODO change the picking process as long with the position included


args = parser.parse_args()

def get_best_model(label_name,initial_position,root_path=args.root_path):
    # [TradeMaster] only low-level runs (beta_*): root_path also holds potential_model/ and high_level/
    coffient_list = [c for c in os.listdir(root_path) if c.startswith("beta_")
                     and os.path.isdir(os.path.join(root_path, c, "seed_{}".format(args.seed)))]
    sort_list(coffient_list)
    result_dict = {}
    for coffient in coffient_list:
        coffient_path = os.path.join(root_path, coffient, "seed_{}".format(args.seed))
        epoch_list = [e for e in os.listdir(coffient_path) if e.startswith("epoch_")]
        sort_list(epoch_list)
        for epoch in epoch_list:
            epoch_path = os.path.join(coffient_path, epoch)
            # [TradeMaster] label_name may be a list (fallback: score over several labels)
            target_paths = [os.path.join(epoch_path, "valid_multi", l, initial_position)
                            for l in (label_name if isinstance(label_name, list) else [label_name])]
            # [TradeMaster] skip epochs/labels that were not validated (valid-low caps epochs per beta)
            target_paths = [t for t in target_paths if os.path.isdir(t)]
            if not target_paths:
                continue
            return_rate_list = []
            normalized_return_rate_list = []
            reward_sum_list=[]
            df_list = []
            for target_path in target_paths:
                names = os.listdir(target_path)
                sort_list(names)
                df_list += [os.path.join(target_path, n) for n in names]
            for df in df_list:
                result_path = df
                return_rate = np.load(
                    os.path.join(result_path, "final_balance.npy"), allow_pickle=True
                ) / (np.load(
                    os.path.join(result_path, "require_money.npy"), allow_pickle=True
                )+1e-12)
                return_rate_list.append(return_rate)
                normalized_return_rate_list.append(
                    return_rate
                    / len(
                        np.load(os.path.join(result_path, "action.npy"), allow_pickle=True)
                    )
                )
                rewards=np.load(
                    os.path.join(result_path, "reward.npy"), allow_pickle=True
                )
                reward_sum_list.append(np.sum(rewards))
            result_dict["{}_{}".format(coffient, epoch)] = {
                "return_rate_list": return_rate_list,
                "normalized_return_rate_list": normalized_return_rate_list,
                "average_return_rate": np.mean(return_rate_list),
                "average_normalized_return_rate": np.mean(normalized_return_rate_list),
                "reward_sum_list":reward_sum_list,
            }


    def find_max_average_return_rate(dictionary):
        max_average_return_rate = float("-inf")
        best_index = list(dictionary.keys())[0]
        for index in dictionary.keys():
            sub_dictionary = dictionary[index]
            average_return_rate = sub_dictionary["average_return_rate"]

            if average_return_rate > max_average_return_rate:
                max_average_return_rate = average_return_rate
                best_index = index

        return best_index
    
    
    def find_max_reward_sum(dictionary):
        max_average_return_rate = float("-inf")
        best_index = list(dictionary.keys())[0]
        for index in dictionary.keys():
            sub_dictionary = dictionary[index]
            average_return_rate = np.mean(sub_dictionary["reward_sum_list"])

            if average_return_rate > max_average_return_rate:
                max_average_return_rate = average_return_rate
                best_index = index

        return best_index


    # [TradeMaster] no result for this label (e.g. the valid set has no segment of this market type):
    # return None instead of silently picking the first key / crashing on an empty directory
    result_dict = {k: v for k, v in result_dict.items() if len(v["reward_sum_list"]) > 0}
    if not result_dict:
        return None
    print(find_max_reward_sum(result_dict))
    return find_max_reward_sum(result_dict)

if __name__=="__main__":
    args = parser.parse_args()
    root_path=args.root_path
    save_path=args.save_path
    initial_position_list=["initial_action_0","initial_action_1","initial_action_2","initial_action_3","initial_action_4"]
    for initial_position in initial_position_list:
        model_path_list=[]
        for i in range(5):
            print("label_{}".format(i))
            index_model=get_best_model(label_name="label_{}".format(i),initial_position=initial_position)
            if index_model is None:  # [TradeMaster]
                print("warning: no validation segments for label_{} / {}; using the best agent over all labels".format(i, initial_position))
                index_model = get_best_model(label_name=["label_{}".format(j) for j in range(5)], initial_position=initial_position)
            if index_model is None:
                raise RuntimeError("no validation results under {} (run valid-low first)".format(root_path))
            elements = index_model.split("_epoch_")
            para=elements[0]
            epoch_number=elements[1]

            model_path_list.append(os.path.join(root_path,para,"seed_{}".format(args.seed),"epoch_{}".format(epoch_number),"trained_model.pkl"))
        model_path_list=list((model_path_list))
        model_list=[torch.load(model_path, map_location="cpu"  # [TradeMaster] GPU-saved checkpoints on a CPU host
        ) for model_path in model_path_list]
        if not os.path.exists(os.path.join(save_path,initial_position)):
            os.makedirs(os.path.join(save_path,initial_position),exist_ok=True)
        for i in range(len(model_list)):
            torch.save(model_list[i],os.path.join(save_path,initial_position,"model_{}.pth".format(i)))