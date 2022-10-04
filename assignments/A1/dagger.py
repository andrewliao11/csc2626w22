import torch
import time
import train_policy
import racer
import argparse
import os
from glob import glob
import json

from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool
import ipdb


def _collect_online_traj(args, driving_policy):
    args.expert_drives = False
    args.out_dir = './dataset/train'
    #args.run_id = dagger_iter
    args.save_expert_actions = len(glob(os.path.join(args.train_dir, f"expert_{args.run_id}*"))) == 0
    args.timesteps = 100000
    
    tot_cross_track_err = racer.run(driving_policy, args)
    return tot_cross_track_err


def _load_trained_policy_if_exists(args, path):
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    _weights = torch.load(path)['model_state_dict']
    driving_policy.load_state_dict(_weights)
    return driving_policy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)
    parser.add_argument("--dagger_iterations", help="", default=10)
    args = parser.parse_args()

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####
    
    print ('TRAINING LEARNER ON INITIAL DATASET')
    args.train_dir = './dataset/train'
    args.validation_dir = './dataset/val'
    args.weights_out_file = f"./weights/learner_0.weights"
    if os.path.exists(args.weights_out_file):
        print('Load weights from previous run ')
        driving_policy = _load_trained_policy_if_exists(args, args.weights_out_file)
    else:
        driving_policy = train_policy.main(args)

    
    res = []
    try:
        for dagger_iter in range(args.dagger_iterations):

            print (f'[{dagger_iter + 1}] GETTING EXPERT DEMONSTRATIONS')
            #if len(glob(os.path.join(args.train_dir, f"expert_{dagger_iter+1}*"))) == 0:
            args.run_id = dagger_iter + 1
            tot_cross_track_err = _collect_online_traj(args, driving_policy)
            res.append(tot_cross_track_err)
            
            print ('RETRAINING LEARNER ON AGGREGATED DATASET')
            args.weights_out_file = f"./weights/learner_{dagger_iter + 1}.weights"
            if os.path.exists(args.weights_out_file):
                driving_policy = _load_trained_policy_if_exists(args, args.weights_out_file)
            else:
                driving_policy = train_policy.main(args)
                

        tot_cross_track_err = _collect_online_traj(args, driving_policy)
        res.append(tot_cross_track_err)
    except:
        pass

    json.dump(res, open('dagger_res', 'w'))
