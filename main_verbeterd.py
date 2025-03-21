import random
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.classic import connect_four_v3

# ========== PettingZoo setup en hulp-functies ==========

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
    return [c for c in range(7) if 0 in board[:, c]]

def drop(board, col, player):
    temp = board.copy()
    row = max(np.where(temp[:, col] == 0)[0])
    temp[row, col] = player
    return temp

def check_win(board, player):
    """Check of 'player' vier op een rij heeft."""
    # Horizontaal
    for r in range(6):
        for c in range(4):
            if all(board[r, c+i] == player for i in range(4)):
                return True
    # Verticaal
    for c in range(7):
        for r in range(3):
            if all(board[r+i, c] == player for i in range(4)):
                return True
    # Diagonaal
    for r in range(3):
        for c in range(4):
            if (all(board[r+i, c+i] == player for i in range(4)) or
                all(board[r+3-i, c+i] == player for i in range(4))):
                return True
    return False

def is_terminal_node(board):
    """Retourneer True als het bord in een terminale toestand is: winst of volle kolommen."""
    if check_win(board, 1) or check_win(board, 2):
        return True
    if all(board[0, c] != 0 for c in range(7)):  # vol bord
        return True
    return False

# ========== Evalueer (score) van een bord ==========

def evaluate_window(window, ai_player):
    """
    'window' is een lijst met 4 cellen (bijv. [board[r,c], board[r,c+1], ...])
    Retourneer een score afhankelijk van hoeveel stenen van de AI (ai_player)
    en van de tegenstander erin zitten.
    """
    opp_player = 3 - ai_player
    score = 0

    ai_count = window.count(ai_player)
    opp_count = window.count(opp_player)
    empty_count = window.count(0)

    # Grote bonus als we 4-op-een-rij hebben
    if ai_count == 4:
        score += 100000
    # Grote straf als de tegenstander 4-op-een-rij heeft
    elif opp_count == 4:
        score -= 100000
    # 3 van AI en 1 leeg => kans om 4 te maken
    elif ai_count == 3 and empty_count == 1:
        score += 50
    # 2 van AI en 2 leeg => kleinere kans
    elif ai_count == 2 and empty_count == 2:
        score += 2
    # 3 van tegenstander en 1 leeg => gevaar
    elif opp_count == 3 and empty_count == 1:
        score -= 40
    # 2 van tegenstander en 2 leeg => mild gevaar
    elif opp_count == 2 and empty_count == 2:
        score -= 2

    return score

def score_position(board, ai_player):
    """Bereken een totale score voor het hele bord."""
    score = 0
    rows, cols = board.shape

    # Maak 'windows' van 4 cellen in elke richting
    # 1) Horizontaal
    for r in range(rows):
        for c in range(cols-3):
            window = list(board[r, c:c+4])
            score += evaluate_window(window, ai_player)
    # 2) Verticaal
    for c in range(cols):
        for r in range(rows-3):
            window = list(board[r:r+4, c])
            score += evaluate_window(window, ai_player)
    # 3) Diagonaal
    for r in range(rows-3):
        for c in range(cols-3):
            window = [board[r+i, c+i] for i in range(4)]
            score += evaluate_window(window, ai_player)
    for r in range(rows-3):
        for c in range(cols-3):
            window = [board[r+3-i, c+i] for i in range(4)]
            score += evaluate_window(window, ai_player)

    return score

# ========== Minimax (met alpha-beta pruning) ==========

def minimax(board, depth, alpha, beta, maximizingPlayer, ai_player):
    """
    - board: huidig bord
    - depth: huidige zoekdiepte
    - alpha, beta: pruning-parameters
    - maximizingPlayer: True (AI aan zet) of False (tegenstander aan zet)
    - ai_player: 1 of 2 (de AI)
    Retourneert (score, kolom)
    """
    valid_cols = valid_moves(board)
    terminal = is_terminal_node(board)

    # Base cases
    if depth == 0 or terminal:
        # Geef finale score of heuristic
        if terminal:
            if check_win(board, ai_player):
                return (999999999, None)
            elif check_win(board, 3 - ai_player):
                return (-999999999, None)
            else:
                return (0, None)  # Gelijkspel of vol bord
        else:
            return (score_position(board, ai_player), None)

    if maximizingPlayer:
        # AI probeert de score te maximaliseren
        value = -float('inf')
        best_col = random.choice(valid_cols)  # fallback
        for col in valid_cols:
            new_board = drop(board, col, ai_player)
            new_score, _ = minimax(new_board, depth-1, alpha, beta, False, ai_player)
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_col
    else:
        # Tegenstander (mens) probeert score te minimaliseren
        value = float('inf')
        best_col = random.choice(valid_cols)
        opp_player = 3 - ai_player
        for col in valid_cols:
            new_board = drop(board, col, opp_player)
            new_score, _ = minimax(new_board, depth-1, alpha, beta, True, ai_player)
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_col

def get_best_move_minimax(board, ai_player, depth=3):
    """
    Zoek de beste kolom via minimax (diepte = 3 standaard).
    Retourneer kolom (int).
    """
    # We zijn zelf 'maximizingPlayer'
    _, col = minimax(board, depth, -float('inf'), float('inf'), True, ai_player)
    return col

# ========== AI-actiefunctie die minimax aanroept ==========

def ai_action_minimax(observation, agent):
    board = observation["board"]
    ai_player = 1 if agent == "player_0" else 2

    # Bepaal beste zet via minimax
    col = get_best_move_minimax(board, ai_player, depth=3)

    return col, "Minimax"

# ========== Hoofd-functie om te spelen ==========

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
            # Menselijke speler
            action = get_human_action(obs_dict)
            rule = "Human"
        else:
            # AI speler
            action, rule = ai_action_minimax(obs_dict, agent)
            print(f"[{agent}] kiest kolom {action} ({rule})")

        env.step(action)
        render_board()

    print("\n" + "="*30)
    render_board()
    print(f"{'Gelijkspel!' if winner is None else f'Speler {winner} wint!'}")
    print("="*30)
    env.close()

if __name__ == "__main__":
    play_game_improved()
