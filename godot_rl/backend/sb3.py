import argparse
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from godot_rl.core.godot_env import GodotEnv
from godot_rl.core.utils import lod_to_dol
import time

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--env_path",
        default=None,
        type=str,
        help="The Godot binary to use, do not include for in editor training",
    )
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="the location of a checkpoint to restore from",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="whether to eval the model",
    )
    parser.add_argument(
        "--speedup",
        default=1,
        type=int,
        help="whether to speed up the physics in the env",
    )
    parser.add_argument(
        "--iterations",
        default=2000000,
        type=int,
        help="How many iterations to run",
    )
    parser.add_argument(
        "--save_every_iterations",
        default=2000000,
        type=int,
        help="After every how many iterations to save model",
    )
    parser.add_argument(
        "--save_path",
        default="_",
        type=str,
        help="Location of save path, example: save/model"
    )
    parser.add_argument(
        "--load_path",
        default="_",
        type=str,
        help="Where to load model from, same as save_path"
    )
    parser.add_argument(
        "--continue_training",
        default="true",
        type=str, 
        help="Whether to continue training or be deterministic (Only if --load_path is specified)"
    )
    parser.add_argument(
        "--log",
        default="true",
        type=str,
        help="Log?"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0003,
        type=float,
        help="Learning Rate"
    )

    return parser.parse_known_args()

def main():
    args, extras = get_args()
    print(args)
    env = StableBaselinesGodotEnv(env_path=args.env_path)
    model = ""

    if args.load_path != "_":
        model = PPO.load(args.load_path, env)
    elif args.save_path != "_":
        if args.log == "true":
            model = PPO(
                    "MultiInputPolicy",
                    env,
                    ent_coef=0.0001,
                    verbose=2,
                    n_steps=32,
                    tensorboard_log=args.save_path,
                    learning_rate=args.learning_rate
                )
        else:
            model = PPO(
                    "MultiInputPolicy",
                    env,
                    ent_coef=0.0001,
                    verbose=2,
                    n_steps=32,
                    learning_rate=args.learning_rate
                )
    else:
        if args.log == "true":
            model = PPO(
                    "MultiInputPolicy",
                    env,
                    ent_coef=0.0001,
                    verbose=2,
                    n_steps=32,
                    tensorboard_log="logs/log",
                    learning_rate=args.learning_rate
                )
        else:
            model = PPO(
                    "MultiInputPolicy",
                    env,
                    ent_coef=0.0001,
                    verbose=2,
                    n_steps=32,
                    learning_rate=args.learning_rate
                )
    
    if args.continue_training == "true":
        t = time.localtime()
        current_time = time.strftime("_%H_%M_%S", t)
        if args.save_path != "_": model.save(args.save_path + current_time)
        for i in range(0, args.iterations, args.save_every_iterations):
            print("Iterations done: " + str(i))
            model.learn(args.save_every_iterations)
            t = time.localtime()
            current_time = time.strftime("_%H_%M_%S", t)
            if args.save_path != "_": model.save(args.save_path + current_time)
    else:
        # env = model.get_env()
        obs = env.reset()
        for i in range(args.iterations):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

    print("closing env")
    env.close()

class StableBaselinesGodotEnv(VecEnv):
    def __init__(self, env_path=None, **kwargs):
        self.env = GodotEnv(env_path=env_path, **kwargs)
        self._check_valid_action_space()

    def _check_valid_action_space(self):
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Tuple):
            assert (
                len(action_space.spaces) == 1
            ), f"sb3 supports a single action space, this env constains multiple spaces {action_space}"

    def _to_tuple_action(self, action):
        return [action]

    def step(self, action):
        action = self._to_tuple_action(action)
        obs, reward, term, trunc, info = self.env.step(action)
        obs = lod_to_dol(obs)

        return {k: np.array(v) for k, v in obs.items()}, np.array(reward), np.array(term), info

    def reset(self):
        obs, info = self.env.reset()
        obs = lod_to_dol(obs)
        obs = {k: np.array(v) for k, v in obs.items()}
        return obs

    def close(self):
        self.env.close()

    def env_is_wrapped(self):
        return [False] * self.env.num_envs

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        # sb3 is not compatible with tuple/dict action spaces
        return self.env.action_space.spaces[0]

    @property
    def num_envs(self):
        return self.env.num_envs

    def env_method(self):
        raise NotImplementedError()

    def get_attr(self):
        raise NotImplementedError()

    def seed(self):
        raise NotImplementedError()

    def set_attr(self):
        raise NotImplementedError()

    def step_async(self):
        raise NotImplementedError()

    def step_wait(self):
        raise NotImplementedError()

if __name__ == "__main__":
    main()