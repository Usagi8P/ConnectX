from statistics import mode
from tkinter.tix import Tree
from turtle import width
from kaggle_environments import evaluate, make, utils
from random import choice

def my_agent(observation, configuration):
    return choice([c for c in range(configuration.columns) if observation.board[c]==0])


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))


def main():
    env = make('connectx', debug=True)
    
    trainer = env.train([None, "random"])
    observation = trainer.reset()
    
    while not env.done:
        my_action = my_agent(observation, env.configuration)
        print("My Action", my_action)
        observation, reward, done, info = trainer.step(my_action)
    
    # Run multiple episodes to estimate its performance.
    print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
    print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

    # This requires jupyter to run
    # env.play(['first', "negamax"], width=500, height=450)


if __name__=='__main__':
    main()
