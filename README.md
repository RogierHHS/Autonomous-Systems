# 🏆 Connect Four - Rule-Based AI Agent

Dit project bevat een **rule-based agent** voor **Connect Four**, geïmplementeerd met de **PettingZoo**-bibliotheek.  
De agent volgt **13 strategische regels** en speelt tegen een menselijke speler.

---

## 📂 Installatie

### 1. **Clone het project**
```bash
git clone https://github.com/RogierHHS/Autonomous-Systems.git
cd Autonomous-Systems

2. Installeer de vereiste libraries

pip install -r requirements.txt

3. Start het spel

python main.py

🎮 Hoe te spelen

    Jij bent speler 1, de AI is speler 2.
    Voer een nummer (0-6) in om een schijf in een kolom te laten vallen.
    De agent reageert en maakt zijn zet.
    Het spel eindigt wanneer een speler vier-op-een-rij krijgt of het bord vol is.

Voorbeeld console-uitvoer:

Beschikbare zetten (nummer = beschikbare kolom):
· · · O X · ·
· · · O X · ·
· · · O X · ·
· · · X · · ·
· · · O · · ·
· · · O · · ·
 0 1 2 3 4 5 6

Kies een kolom (0-6): 4

🧠 AI Strategie

De agent volgt een reeks regels in prioriteitsvolgorde:
✅ Win- en Blokkeerregels (hoogste prioriteit)

    Als de agent een zet kan doen waarmee hij direct vier op een rij krijgt, doe deze zet.
    Als de tegenstander een zet kan doen waarmee hij direct vier op een rij krijgt, blokkeer die zet.

⚔ Aanvalsstrategieën (middelhoge prioriteit)

    Probeer drie op een rij te maken met een open einde.
    Probeer een dubbele dreiging te creëren (twee mogelijke win-situaties tegelijk).

🛡 Verdedigingsstrategieën (middelhoge prioriteit)

    Blokkeer de tegenstander als hij drie op een rij heeft met een open einde.
    Geef de voorkeur aan het spelen in de middelste kolommen boven de buitenste.

📍 Positionele voorkeuren (lage prioriteit)

    Vermijd zetten die de tegenstander een voordeel geven (bijv. een valstrik zetten waarin hij gegarandeerd wint).
    Kies een willekeurige geldige zet als laatste optie.

🛠️ Technische Details

    Programmeertaal: Python
    Bibliotheken:
        pettingzoo → Spelomgeving voor Connect Four
        matplotlib → Grafische weergave van het spelbord
        numpy → Matrixmanipulatie voor bordanalyse
    AI Logica: Rule-based beslissingsproces

📜 Wat zit er in deze README?

✔ Duidelijke installatie-instructies
✔ Gedetailleerde uitleg van de AI-strategie
✔ Uitleg over gebruikte technologieën
✔ Winnaarweergave en voorbeeld-console-uitvoer
