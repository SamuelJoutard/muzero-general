from distutils.command.config import config
import models
from games.splendor import Game, MuZeroConfig
from self_play import MCTS, GameHistory, SelfPlay
import torch
import numpy


game = Game()
config = MuZeroConfig()

# Initialize the network
model = models.MuZeroNetwork(config)
checkpoint = torch.load("C:/Users/sj19/Documents/Splendor/muzero-general/results/splendor/2022-07-09--15-58-21/model.checkpoint")
model.set_weights(checkpoint["weights"])
# model = model.to(torch.device("cuda" if config.selfplay_on_gpu else "cpu"))
model = model.to(torch.device("cuda"))
model.eval()

done = False
temperature_threshold = config.temperature_threshold
temperature = 1.0

game_history = GameHistory()
observation = game.reset()
game_history.action_history.append(0)
game_history.observation_history.append(observation)
game_history.reward_history.append(0)
game_history.to_play_history.append(game.to_play())

with torch.no_grad():
    while (
        not done and game.env.it <= 100
    ):

        assert (
            len(numpy.array(observation).shape) == 3
        ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
        assert (
            numpy.array(observation).shape == config.observation_shape
        ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {config.observation_shape} but got {numpy.array(observation).shape}."
        
        
        
        stacked_observations = game_history.get_stacked_observations(
            -1, config.stacked_observations, len(config.action_space)
        )



        # print(next(model.parameters()).device)
        # print(stacked_observations.device)

        # Choose the action
        root, mcts_info = MCTS(config).run(
            model,
            stacked_observations,
            game.legal_actions(),
            game.to_play(),
            True,
        )
        action = SelfPlay.select_action(
            root,
            temperature
            if not temperature_threshold
            or len(game_history.action_history) < temperature_threshold
            else 0,
        )

        # if render:
        #     print(f'Tree depth: {mcts_info["max_tree_depth"]}')
        #     print(
        #         f"Root value for player {self.game.to_play()}: {root.value():.2f}"
        #     )

        observation, reward, done = game.step(action)

        print(f"Played action: {action}")
        game.render()