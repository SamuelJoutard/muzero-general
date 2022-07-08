import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F


def reward(player, splendor, action):
    """
    0: two blues
    1: two reds
    2: two greens
    3: two browns
    4: two whites
    5: blue red green
    6: blue red brown
    7: blue red white
    8: blue green brown
    9: blue green white
    10: blue brown white
    11: red green brown
    12: red green white
    13: red brown white 
    14: green brown white
    15: reservation 00
    16: reservation 01
    17: reservation 02
    18: reservation 03
    19: reservation 10
    20: reservation 11
    21: reservation 12
    22: reservation 13
    23: reservation 20
    24: reservation 21
    25: reservation 22
    26: reservation 23
    27: pay 00
    28: pay 01
    29: pay 02
    30: pay 03
    31: pay 10
    32: pay 11
    33: pay 12
    34: pay 13
    35: pay 20
    36: pay 21
    37: pay 22
    38: pay 23
    39: pay reservation 0
    40: pay reservation 1
    41: pay reservation 2
    """
    actions_dict = {
        0: ["blue", 0],
        1: ["red", 1],
        2: ["green", 2],
        3: ["brown", 3],
        4: ["white", 4],
        5: [["blue", 0],
            ["red", 1],
            ["green", 2]],
        6: [["blue", 0],
            ["red", 1],
            ["brown", 3]],
        7: [["blue", 0],
            ["red", 1],
            ["white", 4]],
        8: [["blue", 0],
            ["green", 2],
            ["brown", 3]],
        9: [["blue", 0],
            ["green", 2],
            ["white", 4]],
        10: [["blue", 0],
            ["brown", 3],
            ["white", 4]],
        11: [["red", 1],
            ["green", 2],
            ["brown", 3]],
        12: [["red", 1],
            ["green", 2],
            ["white", 4]],
        13: [["red", 1],
            ["brown", 3],
            ["white", 4]],
        14: [["green", 2],
            ["brown", 3],
            ["white", 4]],
        15: [0, 0],
        16: [0, 1],
        17: [0, 2],
        18: [0, 3],
        19: [1, 0],
        20: [1, 1],
        21: [1, 2],
        22: [1, 3],
        23: [2, 0],
        24: [2, 1],
        25: [2, 2],
        26: [2, 3],
        27: [0, 0],
        28: [0, 1],
        29: [0, 2],
        30: [0, 3],
        31: [1, 0],
        32: [1, 1],
        33: [1, 2],
        34: [1, 3],
        35: [2, 0],
        36: [2, 1],
        37: [2, 2],
        38: [2, 3],
        39: 0, 
        40: 1,
        41: 2
    }
    ### Type 0 action: Take twice the same coin
    if action<5:
        if splendor.coins[actions_dict[action][0]]<4:
            r = -50.
            done = True
        elif player.coins_pt.sum()+2>10:
            r = -50.
            done = True
        else:
            splendor.coins[actions_dict[action][0]] -= 2
            splendor.coins_pt[actions_dict[action][1]] -= 2
            player.coins[actions_dict[action][0]] += 2
            player.coins_pt[actions_dict[action][1]] += 2
    ### Type 1 action: Take 3 coins
    elif 5<=action<15:
        if splendor.coins[actions_dict[action][0][0]]<1 or splendor.coins[actions_dict[action][1][0]]<1 or splendor.coins[actions_dict[action][2][0]]<1:
            r = -50.
            done = True
        elif player.coins_pt.sum()+3>10:
            r = -50.
            done = True
        else:
            for i in range(3):
                splendor.coins[actions_dict[action][i][0]] -= 1
                splendor.coins_pt[actions_dict[action][i][1]] -= 1
                player.coins[actions_dict[action][i][0]] += 1
                player.coins_pt[actions_dict[action][i][1]] += 1
            r = -0.1
    ### Type 2 action: Reserve a card
    elif 15<=action<27:
        if player.coins_pt.sum()+1>10:
            r = -50.
            done = True
        elif splendor.coins["gold"]==0:
            r = -50.
            done = True
        elif player.reservation[:11].sum()>0 and player.reservation[11:22].sum()>0 and player.reservation[22:33].sum()>0:
            r = -50.
            done = True
        else:
            splendor.coins["gold"] -= 1
            splendor.coins_pt[5] -= 1
            player.coins["gold"] += 1
            player.coins_pt[5] += 1
            if player.reservation[:11].sum()==0:
                player.reservation[:11] = splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11:(actions_dict[action][1]+1)*11]
            elif player.reservation[11:22].sum()==0:
                player.reservation[11:22] = splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11:(actions_dict[action][1]+1)*11]
            else:
                player.reservation[22:] = splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11:(actions_dict[action][1]+1)*11]
            splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11:(actions_dict[action][1]+1)*11] = splendor.card_decks[str(actions_dict[action][0])][0]
            splendor.card_decks[str(actions_dict[action][0])] = splendor.card_decks[str(actions_dict[action][0])][1:]
            r = -1.
    ### Type 3 action: Buy a card
    elif 27<=action<39:
        tot_bl = player.cards["blue"] + player.coins["blue"]
        tot_r = player.cards["red"] + player.coins["red"]
        tot_g = player.cards["green"] + player.coins["green"]
        tot_b = player.cards["brown"] + player.coins["brown"]
        tot_w = player.cards["white"] + player.coins["white"]
        gdbl = - np.min(0, tot_bl-splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 6])
        gdr = - np.min(0, tot_r-splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 7])
        gdg = - np.min(0, tot_g-splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 8])
        gdb = - np.min(0, tot_b-splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 9])
        gdw = - np.min(0, tot_w-splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 10])
        gold_due =  gdbl + gdr + gdg + gdb + gdw
        if gold_due>player.coins["gold"]:
            r = -50.
            done = True
        else:
            paid_bl = np.min(splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 6].item(), player.coins["blue"])
            paid_r = np.min(splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 7].item(), player.coins["red"])
            paid_g = np.min(splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 8].item(), player.coins["green"])
            paid_b = np.min(splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 9].item(), player.coins["brown"])
            paid_w = np.min(splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 10].item(), player.coins["white"])
            player.coins["blue"] -= paid_bl
            player.coins_pt[0] -= paid_bl
            splendor.coins["blue"] += paid_bl
            splendor.coins_pt[0] += paid_bl
            player.coins["red"] -= paid_r
            player.coins_pt[1] -= paid_r
            splendor.coins["red"] += paid_r
            splendor.coins_pt[1] += paid_r
            player.coins["green"] -= paid_g
            player.coins_pt[2] -= paid_g
            splendor.coins["green"] += paid_g
            splendor.coins_pt[2] += paid_g
            player.coins["brown"] -= paid_b
            player.coins_pt[3] -= paid_b
            splendor.coins["brown"] += paid_b
            splendor.coins_pt[3] += paid_b
            player.coins["white"] -= paid_w
            player.coins_pt[4] -= paid_w
            splendor.coins["white"] += paid_w
            splendor.coins_pt[4] += paid_w
            player.coins["gold"] -= gold_due
            player.coins_pt[5] -= gold_due
            splendor.coins["gold"] += gold_due
            splendor.coins_pt[5] += gold_due

            points = splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11].item()
            player.points += points
            player.points_pt += points

            player.cards["blue"] += splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 1].item()
            player.cards["red"] += splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 2].item()
            player.cards["green"] += splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 3].item()
            player.cards["brown"] += splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 4].item()
            player.cards["white"] += splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 5].item()
            player.cards_pt += splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11 + 1: actions_dict[action][1]*11 + 6]

            splendor.flop[actions_dict[action][0]][actions_dict[action][1]*11:(actions_dict[action][1]+1)*11] = splendor.card_decks[str(actions_dict[action][0])][0]
            splendor.card_decks[str(actions_dict[action][0])] = splendor.card_decks[str(actions_dict[action][0])][1:]

            r = 1. + points
    ### Type 4 action: Buy a reservation
    else:
        tot_bl = player.cards["blue"] + player.coins["blue"]
        tot_r = player.cards["red"] + player.coins["red"]
        tot_g = player.cards["green"] + player.coins["green"]
        tot_b = player.cards["brown"] + player.coins["brown"]
        tot_w = player.cards["white"] + player.coins["white"]
        gdbl = - np.min(0, tot_bl-player.reservations[actions_dict[action]*11+6])
        gdr = - np.min(0, tot_r-player.reservations[actions_dict[action]*11+7])
        gdg = - np.min(0, tot_g-player.reservations[actions_dict[action]*11+8])
        gdb = - np.min(0, tot_b-player.reservations[actions_dict[action]*11+9])
        gdw = - np.min(0, tot_w-player.reservations[actions_dict[action]*11+10])
        gold_due =  gdbl + gdr + gdg + gdb + gdw
        if player.reservations[actions_dict[action]*11:(actions_dict[action]+1)*11].sum()==0:
            r = -50.
            done = True
        elif gold_due>player.coins["gold"]:
            r = -50.
            done = True
        else:
            paid_bl = np.min(player.reservations[actions_dict[action]*11+6].item(), player.coins["blue"])
            paid_r = np.min(player.reservations[actions_dict[action]*11+7].item(), player.coins["red"])
            paid_g = np.min(player.reservations[actions_dict[action]*11+8].item(), player.coins["green"])
            paid_b = np.min(player.reservations[actions_dict[action]*11+9].item(), player.coins["brown"])
            paid_w = np.min(player.reservations[actions_dict[action]*11+10].item(), player.coins["white"])
            player.coins["blue"] -= paid_bl
            player.coins_pt[0] -= paid_bl
            splendor.coins["blue"] += paid_bl
            splendor.coins_pt[0] += paid_bl
            player.coins["red"] -= paid_r
            player.coins_pt[1] -= paid_r
            splendor.coins["red"] += paid_r
            splendor.coins_pt[1] += paid_r
            player.coins["green"] -= paid_g
            player.coins_pt[2] -= paid_g
            splendor.coins["green"] += paid_g
            splendor.coins_pt[2] += paid_g
            player.coins["brown"] -= paid_b
            player.coins_pt[3] -= paid_b
            splendor.coins["brown"] += paid_b
            splendor.coins_pt[3] += paid_b
            player.coins["white"] -= paid_w
            player.coins_pt[4] -= paid_w
            splendor.coins["white"] += paid_w
            splendor.coins_pt[4] += paid_w
            player.coins["gold"] -= gold_due
            player.coins_pt[5] -= gold_due
            splendor.coins["gold"] += gold_due
            splendor.coins_pt[5] += gold_due

            points = player.reservations[actions_dict[action]*11].item()
            player.points += points
            player.points_pt += points

            player.cards["blue"] += player.reservations[actions_dict[action]*11 + 1].item()
            player.cards["red"] += player.reservations[actions_dict[action]*11 + 2].item()
            player.cards["green"] += player.reservations[actions_dict[action]*11 + 3].item()
            player.cards["brown"] += player.reservations[actions_dict[action]*11 + 4].item()
            player.cards["white"] += player.reservations[actions_dict[action]*11 + 5].item()
            player.cards_pt += player.reservations[actions_dict[action]*11 + 1: actions_dict[action]*11 + 6].item()

            player.reservations[actions_dict[action]*11:(actions_dict[action]+1)*11] = 0

            r = 1. + points

    ### To be implemented
    ### Nobles and victory condition
    
    return None



def reward_full_access(player, splendor, action):
    """
    Reward function for Splendor full access mode
    0: two blues
    1: two reds
    2: two greens
    3: two browns
    4: two whites
    5: blue red green
    6: blue red brown
    7: blue red white
    8: blue green brown
    9: blue green white
    10: blue brown white
    11: red green brown
    12: red green white
    13: red brown white 
    14: green brown white
    15: take one gold
    16-55: Buy the corresponding tier 0 card
    56-85: Buy the corresponding tier 1 card
    86-105: Buy the corresponding tier 2 card
    """
    actions_dict = {
        0: ["blue", 0],
        1: ["red", 1],
        2: ["green", 2],
        3: ["brown", 3],
        4: ["white", 4],
        5: [["blue", 0],
            ["red", 1],
            ["green", 2]],
        6: [["blue", 0],
            ["red", 1],
            ["brown", 3]],
        7: [["blue", 0],
            ["red", 1],
            ["white", 4]],
        8: [["blue", 0],
            ["green", 2],
            ["brown", 3]],
        9: [["blue", 0],
            ["green", 2],
            ["white", 4]],
        10: [["blue", 0],
            ["brown", 3],
            ["white", 4]],
        11: [["red", 1],
            ["green", 2],
            ["brown", 3]],
        12: [["red", 1],
            ["green", 2],
            ["white", 4]],
        13: [["red", 1],
            ["brown", 3],
            ["white", 4]],
        14: [["green", 2],
            ["brown", 3],
            ["white", 4]],
        15: ["gold", 5],
        16: 16,
        56: 56,
        86: 86
    }

    done = False

    ### Type 0 action: Take twice the same coin
    if action<5:
        if splendor.coins[actions_dict[action][0]]<4:
            r = -50.
            done = True
        elif player.coins_pt.sum()+2>10:
            r = -50.
            done = True
        else:
            splendor.coins[actions_dict[action][0]] -= 2
            splendor.coins_pt[actions_dict[action][1]] -= 2
            player.coins[actions_dict[action][0]] += 2
            player.coins_pt[actions_dict[action][1]] += 2
            r = 2
    ### Type 1 action: Take 3 coins
    elif 5<=action<15:
        if splendor.coins[actions_dict[action][0][0]]<1 or splendor.coins[actions_dict[action][1][0]]<1 or splendor.coins[actions_dict[action][2][0]]<1:
            r = -50.
            done = True
        elif player.coins_pt.sum()+3>10:
            r = -50.
            done = True
        else:
            for i in range(3):
                splendor.coins[actions_dict[action][i][0]] -= 1
                splendor.coins_pt[actions_dict[action][i][1]] -= 1
                player.coins[actions_dict[action][i][0]] += 1
                player.coins_pt[actions_dict[action][i][1]] += 1
            r = 3
    ### Type 2 action: Reserve a card which in this mode is equivalent to get one gold (because of the full access)
    elif action==15:
        if player.coins_pt.sum()+1>10:
            r = -50.
            done = True
        elif splendor.coins["gold"]==0:
            r = -50.
            done = True
        else:
            splendor.coins["gold"] -= 1
            splendor.coins_pt[5] -= 1
            player.coins["gold"] += 1
            player.coins_pt[5] += 1
            r = 1
    ### Type 3 action: Buy a card
    else:
        if 16<=action<56:
            deck = "0"
            ind = action - 16
        elif 56<=action<86:
            deck = "1"
            ind = action - 56
        elif 86<=action<106:
            deck = "2"
            ind = action - 86
        if splendor.card_availability[deck][ind]==0:
            r = -50
        else:
            tot_bl = player.cards["blue"] + player.coins["blue"]
            tot_r = player.cards["red"] + player.coins["red"]
            tot_g = player.cards["green"] + player.coins["green"]
            tot_b = player.cards["brown"] + player.coins["brown"]
            tot_w = player.cards["white"] + player.coins["white"]

            bl_due = int(splendor.card_decks[deck][ind][6].item())
            r_due = int(splendor.card_decks[deck][ind][7].item())
            g_due = int(splendor.card_decks[deck][ind][8].item())
            b_due = int(splendor.card_decks[deck][ind][9].item())
            w_due = int(splendor.card_decks[deck][ind][10].item())



            gdbl = - min(0, tot_bl-bl_due)
            gdr = - min(0, tot_r-r_due)
            gdg = - min(0, tot_g-g_due)
            gdb = - min(0, tot_b-b_due)
            gdw = - min(0, tot_w-w_due)

            gold_due =  gdbl + gdr + gdg + gdb + gdw

            if gold_due>player.coins["gold"]:
                r = -50.
                done = True
            else:
                paid_bl = max(bl_due - gdbl - player.cards["blue"], 0)
                paid_r = max(r_due - gdr - player.cards["red"], 0)
                paid_g = max(g_due - gdg - player.cards["green"], 0)
                paid_b = max(b_due - gdb - player.cards["brown"], 0)
                paid_w = max(w_due - gdw - player.cards["white"], 0)
                player.coins["blue"] -= paid_bl
                player.coins_pt[0] -= paid_bl
                splendor.coins["blue"] += paid_bl
                splendor.coins_pt[0] += paid_bl
                player.coins["red"] -= paid_r
                player.coins_pt[1] -= paid_r
                splendor.coins["red"] += paid_r
                splendor.coins_pt[1] += paid_r
                player.coins["green"] -= paid_g
                player.coins_pt[2] -= paid_g
                splendor.coins["green"] += paid_g
                splendor.coins_pt[2] += paid_g
                player.coins["brown"] -= paid_b
                player.coins_pt[3] -= paid_b
                splendor.coins["brown"] += paid_b
                splendor.coins_pt[3] += paid_b
                player.coins["white"] -= paid_w
                player.coins_pt[4] -= paid_w
                splendor.coins["white"] += paid_w
                splendor.coins_pt[4] += paid_w
                player.coins["gold"] -= gold_due
                player.coins_pt[5] -= gold_due
                splendor.coins["gold"] += gold_due
                splendor.coins_pt[5] += gold_due

                points = splendor.card_decks[deck][ind][0].item()
                player.points += points
                player.points_pt += points

                player.cards["blue"] += splendor.card_decks[deck][ind][1].item()
                player.cards["red"] += splendor.card_decks[deck][ind][2].item()
                player.cards["green"] += splendor.card_decks[deck][ind][3].item()
                player.cards["brown"] += splendor.card_decks[deck][ind][4].item()
                player.cards["white"] += splendor.card_decks[deck][ind][5].item()
                player.cards_pt += splendor.card_decks[deck][ind][1:6]
                player.cards_possession[deck][ind] = 1

                splendor.card_availability[deck][ind] = 0

                r = points + 5*0.9**splendor.turn

    for n in range(splendor.N_nobles):
        if splendor.nobles_availability[n]==1:
            cond = torch.prod((splendor.nobles - player.cards_pt)<=0)
            if cond==1:
                r += 3
                splendor.nobles_availability[n] = 0
    
    if player.points>=15:
        r += 50
        done = True
    
    splendor.turn += 1
    splendor.turn_pt += 1
    
    return r, done