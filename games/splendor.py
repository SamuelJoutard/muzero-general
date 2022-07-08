"""
This is a very simple form of twenty one. Ace only counts as value 1 not 1 or
11 for simplicity. This means that there is no such thing as a natural or two
card 21. This is a good example of showing how it can provide a good solution
to even luck based games.
"""

import datetime
import pathlib

import numpy as np
import torch

from .abstract_game import AbstractGame

from .Splendor_material.splendor import Splendor_game
from .Splendor_material.player import Player


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = ( 1,1,3*(187)+90+6+10) #90 cards + 6 types of chips, 1 flop and 3 players, reservation (3*90 cards) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(90+90+90+15)) # 90 buy cards, 90 pay reservation, 90 do reservation, 15 chips collection  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(3)) # List of players. You should only edit the length
        self.stacked_observations = 0 # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0 # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 4 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 100 # Maximum number of moves if game is not finished before
        self.num_simulations = 21 # Number of future moves self-simulated
        self.discount = 1 # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 32  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 32  # Number of channels in value head
        self.reduced_channels_policy = 32  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 15000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.03  # Initial learning rate
        self.lr_decay_rate = 0.75  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 150000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Splendor()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(
            f"Enter the action (0) Hit, or (1) Stand for the player {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter either (0) Hit or (1) Stand : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Hit",
            1: "Stand",
        }
        return f"{action_number}. {actions[action_number]}"


class Splendor:
    def __init__(self):
        self.game = Splendor_game()
        self.players = {
            0: Player(),
            1: Player(),
            2: Player()
        }
        self.player = 0

        self.it = 0

        self.coin_action = {
            0 : np.array([2, 0, 0, 0, 0, 0]),
            1 : np.array([0, 2, 0, 0, 0, 0]),
            2 : np.array([0, 0, 2, 0, 0, 0]),
            3 : np.array([0, 0, 0, 2, 0, 0]),
            4 : np.array([0, 0, 0, 0, 2, 0]),
            5 :  np.array([1, 1, 1, 0, 0, 0]),
            6 :  np.array([1, 1, 0, 1, 0, 0]),
            7 :  np.array([1, 1, 0, 0, 1, 0]),
            8 :  np.array([1, 0, 1, 1, 0, 0]),
            9 :  np.array([1, 0, 1, 0, 1, 0]),
            10 : np.array([1, 0, 0, 1, 1, 0]),
            11 : np.array([0, 1, 1, 1, 0, 0]),
            12 : np.array([0, 1, 1, 0, 1, 0]),
            13 : np.array([0, 1, 0, 1, 1, 0]),
            14 : np.array([0, 0, 1, 1, 1, 0])
        }

    def to_play(self):
        return self.it%3

    def reset(self):
        self.game = Splendor_game()
        self.players = {
            0: Player(),
            1: Player(),
            2: Player()
        }
        self.player = 0

        self.n_players = 3

        self.it = 0
        return self.get_observation()

    def step(self, action):
        """
        90 buy cards, 90 pay reservation, 90 do reservation, 15 chips collection
        """
        player = self.players[self.player]
        
        if action<90: # Buy a card
            card_id = action
            check_player = player.check_collect_card(card_id)
            check_splendor = self.game.check_collect_card(card_id)
            if check_player*check_splendor==1:
                self.game.collect_card(card_id)
                player.collect_card(card_id)
                reward = 1 + player.cards[card_id, 0]
                check_noble = self.game.check_nobles(player)
                if check_noble is not None:
                    for noble_idx in check_noble:
                        self.game.collect_noble(noble_idx)
                        reward += 3
                        player.points += 3
            else:
                reward = -3

        elif action<180: # Buy a reservation
            card_id = action - 90
            check_player = player.check_pay_reservation(card_id)
            if check_player==1:
                player.pay_reservation(card_id)
                reward = 1 + player.cards[card_id, 0]
            check_noble = self.game.check_nobles(player)
            if check_noble is not None:
                for noble_idx in check_noble:
                    self.game.collect_noble(noble_idx)
                    reward += 3
                    player.points += 3
            else:
                reward = -3
        
        elif action<270: # Reserve a card
            card_id = action - 180
            check_player = player.check_reserve_card(card_id)
            check_splendor = self.game.check_reserve_card(card_id)
            if check_player*check_splendor==1:
                self.game.reserve_card(card_id)
                player.reserve_card(card_id)
                reward = 1 + player.cards[action, 0]
            else:
                reward = -3
            
        else: # collect coins
            pick_id = action - 270
            coins_to_collect = self.coin_action[pick_id]
            check_player = player.check_collect_coins(coins_to_collect)
            check_splendor = self.game.check_collect_coins(coins_to_collect)
            if check_player*check_splendor==1:
                self.game.collect_coins(coins_to_collect)
                player.collect_coins(coins_to_collect)
                reward = 0
            else:
                reward = -3

        done = player.points>=15
        if done:
            reward += 10

        self.it += 1
        self.player = self.to_play()

        return self.get_observation(), reward, done

    def get_observation(self):
        return np.concatenate([self.players[0].get_state(), self.players[1].get_state(), self.players[2].get_state(), self.game.get_state(), np.array([self.player])], axis=0)
