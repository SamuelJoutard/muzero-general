import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decks_and_nobles import deck0, deck1, deck2, nobles



class Splendor_game(object):
    def __init__(self,):
        # self.coins = {"blue": 5,
        #               "red": 5,
        #               "green": 5,
        #               "brown": 5,
        #               "white": 5,
        #               "gold": 5}
        self.coins = np.zeros(6) + 5

        self.card_decks = {"0": deck0,
                           "1": deck1,
                           "2": deck2}
        self.card_decks_order = {
            "0" : np.random.permutation(40),
            "1" : np.random.permutation(30)+40,
            "2" : np.random.permutation(20)+70
        }

        self.flop = np.zeros(90)
        self.flop[self.card_decks_order["0"][:4]] = 1
        self.flop[self.card_decks_order["1"][:4]] = 1
        self.flop[self.card_decks_order["2"][:4]] = 1
        
        self.card_decks_order["0"] = self.card_decks_order["0"][4:]
        self.card_decks_order["1"] = self.card_decks_order["1"][4:]
        self.card_decks_order["2"] = self.card_decks_order["2"][4:]


        self.nobles = nobles
        self.nobles_game = np.zeros(10)
        self.nobles_game[np.random.permutation(10)[:4]] = 1


    def get_state(self):
        return np.concatenate([self.coins, self.flop, self.nobles_game], axis=0)


    def collect_card(self, number):
        self.flop[number] = 0
        if number<40:
            if len(self.card_decks_order["0"])>0:
                self.flop[self.card_decks_order["0"][0]] = 1
                self.card_decks_order["0"] = self.card_decks_order["0"][1:]
        elif number<70:
            if len(self.card_decks_order["1"])>0:
                self.flop[self.card_decks_order["1"][0]] = 1
                self.card_decks_order["1"] = self.card_decks_order["1"][1:]
        else:
            if len(self.card_decks_order["2"])>0:
                self.flop[self.card_decks_order["2"][0]] = 1
                self.card_decks_order["2"] = self.card_decks_order["2"][1:]

    def check_collect_card(self, number):
        if self.flop[number]==0:
            return 0
        return 1


    def reserve_card(self, number):
        self.collect_card(number)
        self.coins[5] -= 1

    def check_reserve_card(self, number):
        if self.flop[number]==0 or self.coins[5]<1:
            return 0
        return 1 


    def check_nobles(self, player):
        current_colors = (player.cards_bought[:, None] * player.cards[:, 1:6]).sum(0)
        additional_gems_needed = np.maximum(self.nobles - current_colors[None, :], 0).sum(1)
        matching = (additional_gems_needed==0) * self.nobles_game
        idx = np.nonzero(matching)
        if len(idx)>0:
            return idx
        else:
            return None

    def collect_noble(self, number):
        self.nobles_game[number] = 0


    def collect_coins(self, coins_to_collect):
        self.coins -= coins_to_collect

    def check_collect_coins(self, coins_to_collect):
        if coins_to_collect.max()==1:
            if ((self.coins - coins_to_collect)<0).sum()>0:
                return 0
        else:
            idx = np.nonzero(coins_to_collect)[0]
            if self.coins[idx]<3:
                return 0
        return 1

