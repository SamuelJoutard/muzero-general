import numpy as np 
from .decks_and_nobles import *

class Player(object):
    def __init__(self,):
        # self.coins = {"blue": 0,
        #               "red": 0,
        #               "green": 0,
        #               "brown": 0,
        #               "white": 0,
        #               "gold": 0}
        self.coins = np.zeros(6)
        self.cards_bought = np.zeros(90)
        self.cards_reserved = np.zeros(90)
        self.points = 0

        self.cards = np.concatenate([deck0, deck1, deck2], axis=0)


    def collect_coins(self, coins):
        self.coins += coins

    def check_collect_coins(self, coins):
        if (self.coins + coins).sum()>10:
            return 0
        else:
            return 1
        
    
    def collect_card(self, number):
        self.cards_bought[number] = 1
        self.points += self.cards[number, 0]
    
    def check_collect_card(self, number):
        cost = self.cards[number, 6:]
        gold_needed = (np.max(cost - self.coins[:5], 0)).sum()
        if gold_needed<=self.coins[5]:
            return 1
        else:
            return 0

    
    def reserve_card(self, number):
        self.cards_reserved[number] = 1
        self.coins[5] += 1

    def check_reserve_card(self, number):
        if self.cards_reserved.sum()>2:
            return 0
        if self.coins.sum()>9:
            return 0
        return 1


    def pay_reservation(self, number):
        self.cards_reserved[number] = 0
        self.cards_bought[number] = 1
        cost = self.cards[number, 6:]
        gold_to_pay = 0
        for i in range(5):
            gold_to_pay += np.max(0, -self.coins[i]+cost[i])
            self.coins[i] = np.max(0, self.coins[i]-cost[i])
        self.coins[-1] -= gold_to_pay

    def check_pay_reservation(self, number):
        cost = self.cards[number, 6:]
        gold_needed = (np.max(cost - self.coins[:5], 0)).sum()
        if gold_needed<=self.coins[5]:
            return 1
        else:
            return 0
            


    def get_state(self):
        return np.concatenate([
            np.array([self.points]),
            self.cards_bought,
            self.cards_reserved,
            self.coins
        ], axis=0) # 187



