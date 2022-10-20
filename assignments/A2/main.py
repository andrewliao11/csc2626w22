import sys
import pickle
import gym
import gym_thing
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
from dmp import DMP
from policy import BallCatchDMPPolicy
import ipdb


PATH_TO_HOME = str(pathlib.Path(__file__).parent.resolve())
PATH_TO_NLOPTJSON = str((
    pathlib.Path(__file__).parent /
    "ballcatch_env" /
    "gym_thing" /
    "nlopt_optimization" /
    "config" /
    "nlopt_config_stationarybase.json"
).resolve())

TRAINED_DMP_PATH = "results/trained_dmp.pkl"

def make_env(online=True, dataset=None, n_substeps = 1, gravity_factor_std=0., n_substeps_std=0.,
             mass_std = 0., act_gain_std = 0., joint_noise_std=0., vicon_noise_std=0., control_noise_std=0., random_training_index=False):
    # Create thing env
    initial_arm_qpos = np.array([1, -.3, -1, 0, 1.6, 0])
    initial_ball_qvel = np.array([-1, 1.5, 4.5])
    initial_ball_qpos = np.array([1.22, -1.6, 1.2])
    zero_base_qpos = np.array([0.0, 0.0, 0.0])

    training_data = None
    if not online:
        training_data = np.load(dataset)
        print("OFFLINE")

    env = gym.make('ballcatch-v0',
                    model_path="robot_with_cup.xml",
                    nlopt_config_path=PATH_TO_NLOPTJSON,
                    initial_arm_qpos=initial_arm_qpos,
                    initial_ball_qvel=initial_ball_qvel,
                    initial_ball_qpos=initial_ball_qpos,
                    initial_base_qpos=zero_base_qpos,
                    pos_rew_weight=.1,
                    rot_rew_weight=0.01,
                    n_substeps=n_substeps,
                    online=online,
                    training_data=training_data,
                    gravity_factor_std = gravity_factor_std,
                    n_substeps_std = n_substeps_std,
                    mass_std = mass_std,
                    act_gain_std = act_gain_std,
                    joint_noise_std = joint_noise_std,
                    vicon_noise_std = vicon_noise_std,
                    control_noise_std = control_noise_std,
                    random_training_index = random_training_index)
    env.reset()
    return env


def test_policy(eval_env, policy, eval_episodes=5, render_freq=1, seed=1):
    # Set seeds
    eval_env.seed(seed)
    np.random.seed(seed)

    avg_reward = 0.
    successes = 0


    for eps in range(eval_episodes):
        print(f"\rEvaluating Episode {eps+1}/{eval_episodes}", end='')
        state, done = eval_env.reset(), False
        policy.set_goal(state=state, goal=eval_env.env.goal)
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            if render_freq != 0 and eps % render_freq == 0:
                eval_env.render()
                # time.sleep(0.01)
            avg_reward += reward
        if "TimeLimit.truncated" in info:
            actual_done = not info["TimeLimit.truncated"]
        else:
            actual_done = done

        if actual_done:
            successes += 1


    avg_reward /= eval_episodes
    success_pct = float(successes)/eval_episodes

    print("")
    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}, success%: {:.3f}".format(eval_episodes, avg_reward, success_pct))
    print("---------------------------------------")
    print("")
    return avg_reward, success_pct


def load_dataset(dataset_path="data/demos.pkl"):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return [traj[:, :6] for traj in dataset["trajectories"]] # 1st 6 elements are joint angle


def q2_recons():
    trajectories = load_dataset()
    # TODO: Train a DMP on trajectories[0]
    dmp = DMP()

    dt = 0.04
    demo, demo_time = dmp._interpolate([trajectories[0]], dt)
    dmp.learn(demo, demo_time)

    # rollout
    x0 = demo[:, 0, :][:, None, :]              # x0: num_demos, 1, num_dofs
    tau = demo_time[:, -1].reshape(-1, 1)       # tau: num_demos, 1
    g = demo[:, -1, :][:, None, :]              # g: num_demos, 1, num_dofs

    rollout = dmp.rollout(dt, tau, x0, g)
    rollout_time = demo_time
    
    for k in range(6):
        plt.figure()
        plt.plot(demo_time[0], demo[0][:, k], label='GT')
        plt.plot(rollout_time[0], rollout[0][:-1, k], label='DMP')
        plt.legend()
        plt.savefig(f'results/recons_{k}.png')


def q2_tuning():
    trajectories = load_dataset()

    # TODO: select the best settings for fitting the demos
    best_config = dict(
        err=np.inf, 
        dmp=None, 
        nbasis=None, 
        k=None
    )
    for nbasis in [2, 10, 20, 30]:
        for k in [1, 10, 50, 100, 200, 250]:
            print("="*50)
            print(f"Try: nbasis={nbasis}, k={k}")
            dmp = DMP(nbasis=nbasis, K_vec=k*np.ones((6,)))
            demo, demo_time = dmp._interpolate(trajectories, 0.04)
            dmp.learn(demo, demo_time)

            rollout = []
            for demo_i, demo_time_i in zip(demo, demo_time):
                x0 = demo_i[0, :][None, None, :]              # x0: num_demos, 1, num_dofs
                tau = demo_time_i[-1].reshape(-1, 1)       # tau: num_demos, 1
                g = demo_i[-1, :][None, None, :]              # g: num_demos, 1, num_dofs

                dt = demo_time_i[1] - demo_time_i[0]
                rollout_i = dmp.rollout(dt, tau, x0, g)[0][:len(demo_i), :]
                rollout.append(rollout_i)

            rollout = np.stack(rollout, 0)
            err = ((rollout - demo) ** 2).mean() ** 0.5
            if err < best_config["err"]:
                best_config["nbasis"] = nbasis
                best_config["k"] = k
                best_config["err"] = err
                best_config["dmp"] = dmp
                print(f"Find a better config (nbasis={best_config['nbasis']}, k={best_config['k']}) with err: {err}")
    
    print(f"Best config: {best_config}")        
    dmp.save(TRAINED_DMP_PATH)


def main():
    env = make_env(n_substeps=1)
    dmp = DMP.load(TRAINED_DMP_PATH)

    policy = BallCatchDMPPolicy(dmp, dt=env.dt)
    test_policy(env, policy, eval_episodes=20, render_freq=1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'recons':
        q2_recons()
    elif sys.argv[1] == 'tuning':
        q2_tuning()
