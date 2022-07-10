import numpy as np 
from .decks_and_nobles import *

from .viz_tools import *

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
        cost = self.cards[number][6:]
        gold_to_pay = 0
        paid = [0]*6
        for i in range(5):
            paid[i] = min(cost[i], self.coins[i])
            gold_to_pay += np.maximum(0, -self.coins[i]+cost[i])
            self.coins[i] = np.maximum(0, self.coins[i]-cost[i])
        self.coins[-1] -= gold_to_pay
        paid[5] = gold_to_pay
        return paid

    
    def check_collect_card(self, number):
        cost = self.cards[number, 6:]
        gold_needed = (np.maximum(cost - self.coins[:5], 0)).sum()
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
        paid = np.zeros(6)
        for i in range(5):
            gold_to_pay += np.maximum(0, -self.coins[i]+cost[i])
            self.coins[i] = np.maximum(0, self.coins[i]-cost[i])
            paid[i] = min(cost[i], self.coins[i])
        self.coins[-1] -= gold_to_pay
        paid[5] = gold_to_pay
        
        return paid

    def check_pay_reservation(self, number):
        cost = self.cards[number, 6:]
        gold_needed = (np.maximum(cost - self.coins[:5], 0)).sum()
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
    
    def get_frame(self):
        coin_img = draw_coins(self.coins)

        cards_id = np.nonzero(self.cards_bought)[0]
        bought = []
        for i, id in enumerate(cards_id):
            bought_img = draw_card(self.cards[id])
            bought.append(bought_img)

        if len(bought)>0:
            bought_im = cat_with_sep(*bought, axis_to_cat=1, line_value=0.)
        else:
            bought_im = np.zeros((21, 18, 3))

        cards_id = np.nonzero(self.cards_reserved)[0]
        reserved = []
        for i, id in enumerate(cards_id):
            reserved_img = draw_card(self.cards[id])
            reserved.append(reserved_img)

        if len(reserved)>0:
            reserved_im = cat_with_sep(*reserved, axis_to_cat=1, line_value=0.)
        else:
            reserved_im = np.zeros((21, 18, 3))

        cards_im = cat_with_sep(*[reserved_im, bought_im], axis_to_cat=0, line_value=0.)

        points = self.points

        points_im = np.zeros((32, 3, 3))
        for p in np.arange(points):
            points_im[1+2*p, 1] = np.array([249./255, 29./255, 227./255])

        return cat_and_pad(*[coin_img, cards_im, points_im], axis_to_pad=0, axis_to_cat=1)


