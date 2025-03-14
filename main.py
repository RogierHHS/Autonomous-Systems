import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.classic import connect_four_v3

# Initialiseer de PettingZoo-omgeving
env = connect_four_v3.env(render_mode="rgb_array")
env.reset(seed=42)

def convert_observation(obs):
    """
    Zet de PettingZoo Connect Four observatie om naar een 6x7 bord.

    Args:
        obs (numpy.ndarray): 3D-array (6x7x2) waarin de eerste laag speler 1 (X) en 
                             de tweede laag speler 2 (O) vertegenwoordigt.

    Returns:
        numpy.ndarray: 2D-array (6x7) met waarden:
                       - 0 = lege positie
                       - 1 = speler 1 (X)
                       - 2 = speler 2 (O)
    """
    board = np.zeros((6, 7), dtype=int)
    for i in range(6):
        for j in range(7):
            if obs[i, j, 0] == 1:
                board[i, j] = 1
            elif obs[i, j, 1] == 1:
                board[i, j] = 2
    return np.flipud(board)  # Omkeren zodat de onderste rij correct wordt weergegeven

def render_board():
    """
    Visualiseert het Connect Four bord met Matplotlib.
    Toont een afbeelding van de huidige staat van het spelbord.
    """
    rgb_image = env.render()
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

def get_human_action(observation):
    """
    Vraagt de menselijke speler om een zet en valideert de invoer.

    Args:
        observation (dict): Bevat het huidige bord ('board') en de mogelijke zetten ('action_mask').

    Returns:
        int: De door de speler gekozen kolom (0-6).
    """
    mask = observation["action_mask"]
    board = observation["board"]

    print("\nBeschikbare zetten (nummer = beschikbare kolom):")
    visual_board = np.copy(board).astype(str)

    for i in range(6):
        for j in range(7):
            if visual_board[i, j] == "0":
                visual_board[i, j] = "·"
            elif visual_board[i, j] == "1":
                visual_board[i, j] = "X"
            else:
                visual_board[i, j] = "O"

    print("\n".join([" ".join(row) for row in np.flipud(visual_board)]))
    print(" 0 1 2 3 4 5 6")

    while True:
        try:
            move = int(input("\nKies een kolom (0-6): "))
            if 0 <= move < 7 and mask[move]:
                return move
            print("Ongeldige zet, kies een van de beschikbare kolommen.")
        except ValueError:
            print("Voer een geldig nummer in (0-6).")

def rule_based_action(observation, agent):
    """
    Selecteert een zet voor de AI op basis van een regelgebaseerde strategie.

    Regels:
        1. Direct winnen als mogelijk.
        2. Blokkeer de tegenstander als hij kan winnen.
        3. Speel een zet die toekomstige winstmogelijkheden creëert.
        4. Probeer een dubbele dreiging te creëren.
        5. Blokkeer tegenstander als hij drie-op-een-rij heeft met een open einde.
        6. Geef de voorkeur aan de middelste kolommen.
        7. Vermijd zetten die de tegenstander een voordeel geven.
        8. Kies een willekeurige geldige zet als laatste optie.

    Args:
        observation (dict): Bevat het huidige bord ('board') en de mogelijke zetten ('action_mask').
        agent (str): Geeft aan of het 'player_1' of 'player_2' is.

    Returns:
        int: De gekozen kolom (0-6) waarin de AI zijn zet plaatst.
    """
    mask = observation["action_mask"]
    board = observation["board"]
    player = 1 if agent == "player_1" else 2
    opponent = 2 if player == 1 else 1

    valid_moves = [col for col in range(7) if mask[col]]

    if not valid_moves:
        return None  # Geen geldige zetten meer

    # Regel 1: Direct winnen als mogelijk
    for col in valid_moves:
        temp_board = np.copy(board)
        row = max(np.where(temp_board[:, col] == 0)[0])  # Kiest de onderste lege rij
        temp_board[row, col] = player
        if check_win(temp_board, player):
            return col  

    # Regel 2: Blokkeer de tegenstander als hij kan winnen
    for col in valid_moves:
        temp_board = np.copy(board)
        row = max(np.where(temp_board[:, col] == 0)[0])
        temp_board[row, col] = opponent
        if check_win(temp_board, opponent):
            return col  

    # Regel 3: Speel een zet die toekomstige winstmogelijkheden creëert
    for col in valid_moves:
        temp_board = np.copy(board)
        row = max(np.where(temp_board[:, col] == 0)[0])
        temp_board[row, col] = player
        if count_three_in_a_row(temp_board, player) > 0:
            return col

    # Regel 4: Probeer een dubbele dreiging te creëren
    for col in valid_moves:
        temp_board = np.copy(board)
        row = max(np.where(temp_board[:, col] == 0)[0])
        temp_board[row, col] = player
        if count_double_threat(temp_board, player):
            return col

    # Regel 5: Blokkeer tegenstander als hij drie-op-een-rij heeft met een open einde
    for col in valid_moves:
        temp_board = np.copy(board)
        row = max(np.where(temp_board[:, col] == 0)[0])
        temp_board[row, col] = opponent
        if count_three_in_a_row(temp_board, opponent) > 0:
            return col

    # Regel 6: Geef de voorkeur aan de middelste kolommen
    middle_moves = [3, 2, 4, 1, 5]  
    for col in middle_moves:
        if col in valid_moves:
            return col

    # Regel 7: Vermijd zetten die de tegenstander een voordeel geven
    safe_moves = [col for col in valid_moves if not check_losing_move(board, col, player)]
    if safe_moves:
        return np.random.choice(safe_moves)

    # Regel 8: Kies een willekeurige geldige zet als laatste optie
    return np.random.choice(valid_moves)

def count_three_in_a_row(board, player):
    """
    Controleert hoeveel keer de speler drie-op-een-rij heeft met een open einde.

    Een open einde betekent dat de rij van drie niet direct wordt geblokkeerd door een andere steen,
    zodat de speler een kans heeft om vier-op-een-rij te maken.

    Args:
        board (numpy.ndarray): 2D-array (6x7) die de huidige staat van het bord vertegenwoordigt.
        player (int): 1 voor speler 1 (X), 2 voor speler 2 (O).

    Returns:
        int: Het aantal keren dat de speler drie-op-een-rij heeft met een open einde.
    """
    return sum(
        sum(board[r, c + i] == player for i in range(4)) == 3 and 0 in board[r, c:c + 4]
        for r in range(6) for c in range(7 - 3)
    )

def count_double_threat(board, player):
    """
    Controleert of een zet twee dreigingen tegelijk creëert.

    Een dubbele dreiging betekent dat de speler een zet kan doen waardoor er twee mogelijke manieren ontstaan
    om in de volgende beurt vier-op-een-rij te krijgen.

    Args:
        board (numpy.ndarray): 2D-array (6x7) die de huidige staat van het bord vertegenwoordigt.
        player (int): 1 voor speler 1 (X), 2 voor speler 2 (O).

    Returns:
        bool: True als de speler een dubbele dreiging heeft, anders False.
    """
    return count_three_in_a_row(board, player) >= 2

def check_losing_move(board, col, player):
    """
    Controleert of een zet de tegenstander direct laat winnen.

    Deze functie simuleert het plaatsen van een schijf in de gegeven kolom en controleert
    of de tegenstander in de volgende beurt direct kan winnen.

    Args:
        board (numpy.ndarray): 2D-array (6x7) die de huidige staat van het bord vertegenwoordigt.
        col (int): De kolom waarin de speler overweegt een zet te plaatsen.
        player (int): 1 voor speler 1 (X), 2 voor speler 2 (O).

    Returns:
        bool: True als de tegenstander in de volgende beurt kan winnen, anders False.
    """
    temp_board = np.copy(board)
    row = np.where(temp_board[:, col] == 0)[0][0]
    temp_board[row, col] = player
    return check_win(temp_board, 3 - player)

def check_win(board, player):
    """
    Controleert of een speler vier-op-een-rij heeft in het huidige bord.

    Dit wordt gecontroleerd voor:
    - Horizontale rijen
    - Verticale kolommen
    - Diagonale lijnen

    Args:
        board (numpy.ndarray): 2D-array (6x7) die de huidige staat van het bord vertegenwoordigt.
        player (int): 1 voor speler 1 (X), 2 voor speler 2 (O).

    Returns:
        bool: True als de speler vier-op-een-rij heeft, anders False.
    """
    for r in range(6):
        for c in range(7 - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    for r in range(6 - 3):
        for c in range(7):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    return False

def play_game():
    """
    Start en beheert een Connect Four-spel tussen een menselijke speler en de AI.

    Voert het spel uit met beurtwisselingen en bepaalt de winnaar.
    """
    env.reset(seed=42)
    render_board()

    winner = None  # Sla de winnaar op

    for agent in env.agent_iter():
        raw_observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            winner = 1 if agent == "player_1" else 2  # Bepaal de winnaar
            break  # Stop de spel-loop
        else:
            observation = {"board": convert_observation(raw_observation["observation"]), "action_mask": raw_observation["action_mask"]}
            action = get_human_action(observation) if agent == "player_1" else rule_based_action(observation, agent)
            env.step(action)
            render_board()

    # Laatste render van het winnende bord en print winnaar
    print("\n" + "="*30)
    render_board()
    if winner:
        print(f" Speler {winner} wint! ")
    else:
        print(" Gelijkspel!")
    print("="*30)

    env.close()

# Zorg ervoor dat het spel alleen start als het script direct wordt uitgevoerd
if __name__ == "__main__":
    play_game()