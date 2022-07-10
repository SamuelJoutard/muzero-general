import numpy as np

# blue, red, green, brown, white, gold

COLORS = {
    0 : np.pad(np.array([[[0., 0., 1.]]]), ((1, 1), (1, 1), (0, 0)), mode="edge"),
    1 : np.pad(np.array([[[1., 0., 0.]]]), ((1, 1), (1, 1), (0, 0)), mode="edge"),
    2 : np.pad(np.array([[[0., 1., 0.]]]), ((1, 1), (1, 1), (0, 0)), mode="edge"),
    3 : np.pad(np.array([[[101./255, 26./255, 36./255]]]), ((1, 1), (1, 1), (0, 0)), mode="edge"),
    4 : np.pad(np.array([[[1., 1., 1.]]]), ((1, 1), (1, 1), (0, 0)), mode="edge"),
    5 : np.pad(np.array([[[242./255, 249./255, 29./255]]]), ((1, 1), (1, 1), (0, 0)), mode="edge")
}


def cat_and_pad(*args, axis_to_pad, axis_to_cat):
    """
    concatenate along one axis and padds the other one
    *args: list of 2D images
    """
    args = list(args)

    W = np.max([x.shape[axis_to_pad] for x in args])
    for i, arr in enumerate(args):
        pad = np.array(((0, 0), (0, 0), (0, 0)))
        pad[axis_to_pad][0] = W - arr.shape[axis_to_pad]
        pad = tuple([tuple(x) for x in pad])
        arr = np.pad(arr, pad)
        args[i] = arr

    return np.concatenate(args, axis=axis_to_cat)


def cat_with_sep(*args, axis_to_cat, line_width=5, line_value=0):
    args = list(args)
    x, y, _ = args[0].shape
    sep = np.zeros((line_width, y, 3))+line_value if axis_to_cat==0 else np.zeros((x, line_width, 3))+line_value


    args_sep = [args[0]] 
    for a in args[1:]:
        args_sep += [sep, a] 

    return np.concatenate(args_sep, axis=axis_to_cat)


def draw_coins(coins):
    coins_images = [None]*6
    for i in range(6):
        if coins[i]>0:
            coins_images[i] = cat_with_sep(*[np.pad(COLORS[i], ((1, 1), (1, 1), (0, 0))) for n in np.arange(coins[i])], axis_to_cat=0)
        else:
            coins_images[i] = np.zeros((5, 5, 3))
    return cat_and_pad(*coins_images, axis_to_pad=0, axis_to_cat=1)


def draw_card(card):
    card = card.astype(int)
    points = card[0]
    color = np.nonzero(card[1:6])[0][0]
    cost = card[6:]

    card_im = np.zeros((21, 18, 3))
    for p in np.arange(points):
        card_im[4, 3+2*p] = np.array([249./255, 29./255, 227./255])
    card_im[4, -5] = COLORS[color][1, 1]

    for i in range(5):
        for p in np.arange(cost[i]):
            card_im[9+2*i, 4 + 2*p] = COLORS[i][1, 1]

    return card_im

def draw_noble(noble):
    noble_im = np.zeros((11, 11, 3))
    for i in range(5):
        for p in np.arange(noble[i]):
            noble_im[1+2*i, 1+2*p] = COLORS[i][1, 1]
    return noble_im
