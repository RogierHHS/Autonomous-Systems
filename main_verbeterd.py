import random
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.classic import connect_four_v3

# Initialiseer de PettingZoo-omgeving
env = connect_four_v3.env(render_mode="rgb_array")

def convert_observation(obs):
    board = np.zeros((6,7), dtype=int)
    for i in range(6):
        for j in range(7):
            if obs[i,j,0] == 1:
                board[i,j] = 1
            elif obs[i,j,1] == 1:
                board[i,j] = 2
    return np.flipud(board)

def render_board():
    plt.imshow(env.render())
    plt.axis('off')
    plt.show()

def get_human_action(observation):
    mask = observation["action_mask"]
    board = observation["board"]
    visual = np.where(board==0, "·", np.where(board==1, "X", "O"))
    print("\n" + "\n".join(" ".join(row) for row in np.flipud(visual)))
    print(" 0 1 2 3 4 5 6")
    while True:
        try:
            move = int(input("Kies kolom (0–6): "))
            if 0 <= move < 7 and mask[move]:
                return move
        except:
            pass
        print("Ongeldige zet — probeer opnieuw.")

def valid_moves(board):
    return [c for c in range(7) if board[0,c] == 0]

def drop(board, col, player):
    temp = board.copy()
    row = max(np.where(temp[:,col] == 0)[0])
    temp[row,col] = player
    return temp

def check_win(board, player):
    for r in range(6):
        for c in range(4):
            if all(board[r,c+i] == player for i in range(4)): return True
    for c in range(7):
        for r in range(3):
            if all(board[r+i,c] == player for i in range(4)): return True
    for r in range(3):
        for c in range(4):
            if all(board[r+i,c+i] == player for i in range(4)) or all(board[r+3-i,c+i] == player for i in range(4)):
                return True
    return False

def rule_based_action_improved(observation, agent):
    board = observation["board"]
    player = 1 if agent == "player_0" else 2
    opponent = 3 - player
    valid = valid_moves(board)

    # Regel 1: Direct winnen
    for col in valid:
        if check_win(drop(board,col,player), player):
            return col, 1

    # Regel 2: Blokkeer tegenstander
    for col in valid:
        if check_win(drop(board,col,opponent), opponent):
            return col, 2

    # Regel 3: Dubbele dreiging creëren
    for col in valid:
        temp = drop(board,col,player)
        if sum(check_win(drop(temp,c,player), player) for c in valid_moves(temp)) >= 2:
            return col, 3

    # Regel 4: Open drie‑in‑een‑rij benutten
    for col in valid:
        temp = drop(board,col,player)
        for r in range(6):
            for c in range(4):
                window = temp[r,c:c+4]
                if list(window).count(player) == 3 and 0 in window:
                    return col, 4

    # Regel 5: Voorkeur midden
    for col in [3,2,4,1,5,0,6]:
        if col in valid:
            return col, 5

    # Regel 6: Vermijd directe verlieszet
    safe = [c for c in valid if not check_win(drop(board,c,opponent), opponent)]
    if safe:
        return random.choice(safe), 6

    # Regel 7: Willekeurige zet
    return random.choice(valid), 7

def play_game_improved():
    env.reset(seed=42)
    render_board()
    winner = None
    agent_to_player = {"player_0":1, "player_1":2}

    for agent in env.agent_iter():
        obs, reward, termination, trunc, _ = env.last()
        if termination or trunc:
            if reward == 1:
                winner = agent_to_player[agent]
            elif reward == -1:
                winner = 3 - agent_to_player[agent]
            break

        obs_dict = {
            "board": convert_observation(obs["observation"]),
            "action_mask": obs["action_mask"]
        }
        if agent == "player_0":
            action = get_human_action(obs_dict)
        else:
            action, _ = rule_based_action_improved(obs_dict, agent)

        env.step(action)
        render_board()

    print("\n" + "="*30)
    render_board()
    print(f"{'Gelijkspel!' if winner is None else f'Speler {winner} wint!'}")
    print("="*30)
    env.close()

if __name__ == "__main__":
    play_game_improved()
