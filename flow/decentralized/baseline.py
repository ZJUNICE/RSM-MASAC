# import the experiment-specific parameters from flow.benchmarks
from flow.benchmarks.grid1 import flow_params

# import the make_create_env to register the environment with OpenAI gym
from flow.utils.registry import make_create_env

import matplotlib.pyplot as plt
from matplotlib import cm, axes
import numpy as np


def main():
    create_env, env_name = make_create_env(flow_params, version=0)
    env = create_env()

    # hyper parameters
    horizon = 800
    num_steps = 200
    num_epochs = 10

    score = np.zeros(25)
    training_score = []
    print_interval = 10
    training_interval = []
    score_interval = []

    # begin simulation
    for n_step in range(num_epochs):
        # initialize environment
        obs = env.reset()
        done = False
        # epoch
        while not done:
            # forward step
            for step in range(num_steps):
                obs_prime, reward, done, _ = env.step(None)
                obs = obs_prime
                score += reward / horizon
                if done:
                    training_score.append(score)
                    score = np.zeros(25)
                    break

        # recording
        if n_step % print_interval == 0 and n_step != 0:
            print('# of epoch : {}, avg score: {}'.format(n_step, training_score[-1].mean()))
            training_interval.append(n_step)
            score_interval.append(training_score[-1].mean())

    np.save('baseline_training_score.npy', training_score)
    np.save('training_interval.npy', training_interval)
    np.save('baseline.npy', score_interval)

    # # for plot
    # plt.figure(1)
    # plt.plot(training_interval, score_interval, 'k.-', linewidth=0.4)
    # plt.title('Training Performance')
    # plt.xlabel('Epochs')
    # plt.ylabel('ARH')
    # plt.axis([0, num_epochs, 0, 1])
    # plt.show()


if __name__ == "__main__":
    main()
