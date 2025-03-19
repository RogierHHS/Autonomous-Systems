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

def rule_based_action(observation, agent):
    mask = observation["action_mask"]
    board = observation["board"]
    player = 1 if agent == "player_0" else 2
    opponent = 3 - player
    valid_moves = [c for c in range(7) if mask[c]]

    if not valid_moves:
        return None, None

    # Regel 1: Direct winnen
    for col in valid_moves:
        temp = board.copy()
        row = max(np.where(temp[:, col] == 0)[0])
        temp[row, col] = player
        if check_win(temp, player):
            return col, 1

    # Regel 2: Blokkeer tegenstander
    for col in valid_moves:
        temp = board.copy()
        row = max(np.where(temp[:, col] == 0)[0])
        temp[row, col] = opponent
        if check_win(temp, opponent):
            return col, 2

    # Regel 3: Creëer drie-op-een-rij
    for col in valid_moves:
        temp = board.copy()
        row = max(np.where(temp[:, col] == 0)[0])
        temp[row, col] = player
        if count_three_in_a_row(temp, player) > 0:
            return col, 3

    # Regel 4: Dubbele dreiging
    for col in valid_moves:
        temp = board.copy()
        row = max(np.where(temp[:, col] == 0)[0])
        temp[row, col] = player
        if count_double_threat(temp, player):
            return col, 4

    # Regel 5: Blokkeer drie-op-een-rij van tegenstander
    for col in valid_moves:
        temp = board.copy()
        row = max(np.where(temp[:, col] == 0)[0])
        temp[row, col] = opponent
        if count_three_in_a_row(temp, opponent) > 0:
            return col, 5

    # Regel 6: Voorkeur midden
    for col in [3, 2, 4, 1, 5]:
        if col in valid_moves:
            return col, 6

    # Regel 7: Vermijd zetten die verlies opleveren
    safe_moves = [c for c in valid_moves if not check_losing_move(board, c, player)]
    if safe_moves:
        return random.choice(safe_moves), 7

    # Regel 8: Willekeurige zet
    return random.choice(valid_moves), 8

def count_three_in_a_row(board, player):
    return sum(
        sum(board[r, c+i] == player for i in range(4)) == 3 and 0 in board[r, c:c+4]
        for r in range(6) for c in range(4)
    )

def count_double_threat(board, player):
    return count_three_in_a_row(board, player) >= 2

def check_losing_move(board, col, player):
    temp = board.copy()
    row = max(np.where(temp[:, col] == 0)[0])
    temp[row, col] = player
    return check_win(temp, 3-player)

def check_win(board, player):
    for r in range(6):
        for c in range(4):
            if all(board[r, c+i] == player for i in range(4)):
                return True
    for c in range(7):
        for r in range(3):
            if all(board[r+i, c] == player for i in range(4)):
                return True
    for r in range(3):
        for c in range(4):
            if all(board[r+i, c+i] == player for i in range(4)) or all(board[r+3-i, c+i] == player for i in range(4)):
                return True
    return False

def play_game():
    env.reset(seed=42)
    render_board()

    winner = None
    agent_to_player = {"player_0":1, "player_1":2}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, _ = env.last()
        if termination or truncation:
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
            action, _ = rule_based_action(obs_dict, agent)

        env.step(action)
        render_board()

    print("\n" + "="*30)
    render_board()
    print(f"{'Gelijkspel!' if winner is None else f'Speler {winner} wint!'}")
    print("="*30)
    env.close()

if __name__ == "__main__":
    play_game()
