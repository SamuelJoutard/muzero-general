import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer

from muzero import MuZero

muzero = MuZero("splendor")

muzero.load_model("C:/Users/sj19/Documents/Splendor/muzero-general/results/splendor/2022-07-09--15-58-21/model.checkpoint")

muzero.test(render=True, opponent="self", muzero_player=None)