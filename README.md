# Kleinwasserkraftwerk Töss - Gymnasium Environment

## Projektbeschreibung
Diese Gymnasium-Environment simuliert ein Laufwasserkraftwerk mit mehreren Turbinen und By-Passes. Es wurde speziell entwickelt, um Reinforcement-Learning-Agenten zu trainieren.

Dieser Python-Code ist im Rahmen eines Reinforcement Learning-Projektes entstanden. Als Inspiration für die Regelstrecke, implementiert als [Gymnasium-Environment](https://gymnasium.farama.org), dient das zweistufige Kleinwasserkraftwerk auf dem Gelände der Rieter Maschinenfabrik in Winterthur, [Obertöss und Niedertöss](https://de.wikipedia.org/wiki/Laufwasserkraftwerke_an_der_Töss). 

## Features

- Hydraulisches Modell basierend auf realistischen Parametern
- Multiple Steuerungsoptionen für Turbinen und By-Passes
- Rendering der Regelstrecke mit pygame zur Visualisierung
- Belohnungsmodell für Stromproduktion und Kostenmanagement

## Setup & Installation
**1. Klonen des Repositories:**
   ```bash
   git clone https://github.com/meic42/agenticLWK.git
   cd agenticLWK
   ```

**2. Installieren der erforderlichen Python-Bibliotheken:**
   ```bash
   pip install -r requirements.txt
   ```
## Let's Play

Ziel der Umgebung ist die Maximierung der Stromproduktion durch optimale Steuerung der Durchflussmengen durch vier Turbinen und zwei By-Pässe. Die Environment kann mit dem Befehl `python play.py` gestartet werden.

![<img src="/img/letsplay.jpg" width="20"/>](/img/letsplay.jpg)

Die Aktoren können mittels Tastatur betätigt werden.

| Taste  | Aktion                     |
|--------|----------------------------|
| w / s  | Turbine 1 auf / zu         |
| e / d  | Turbine 2 auf / zu         |
| r / f  | Bypass 5 auf / zu          |
| u / j  | Turbine 3 auf / zu         |
| i / k  | Turbine 4 auf / zu         |
| o / l  | Bypass 6 auf / zu          |

### Beschreibung der Anlage

Das Stauwehr Obertöss (OT) bildet ein Wasserreservoir, aus dem die zwei Turbinen (Index 1 und 2) der ersten Kraftwerkstufe gespiesen werden. Der Abfluss aus den Turbinen 1 und 2 strömt in den ein zweites Reservoir aus, den Kanal Niedertöss (NT). Aus diesem Reservoir werden die zwei Turbinen (Index 3 und 4) der zweiten Kraftwerkstufe gespiesen. Nach der zweiten Kraftwerkstufe fliesst die entnommene Wassermenge zurück in die Töss.

Der Durchfluss durch die Turbinen 1-4 kann individuell durch die Stellung der Leitwerke (Öffnung 0...100%) eingestellt werden. Die Turbinen der jeweiligen Kraftwerkstufe sind parallel angeordnet und können unabhängig voneinander betrieben werden. Ab einer hydraulischen Leistung von rund 30% des Nominalwertes einer Turbine wird elektrische Energie produziert. 

Um im Betrieb mehr Flexibilität zu haben, können die beiden Kraftwerkstufen umfahren werden. Parallel zu den Turbinen 1 und 2 kann über einen By-Pass (Index 5) Wasser aus dem Stauwehr OT direkt in den Kanal NT geleitet werden. Über einen weiteren By-Pass (Index 6) kann Wasser aus dem Kanal NT unter Umgehung der Turbinen 3 und 4 direkt zurück in die Töss geleitet werden.
Der Durchfluss durch die By-Pässe kann durch deren Schieberstellung (Öffnung 0...100%) eingestellt werden.

### Spielregeln

Die Wasserentnahme aus den Reservoirs erfolgt 10cm unter der Wehrkante. Wird einem Reservoir zu viel Wasser entnommen, respektive führt die Töss wenig Wasser, sinkt der Pegel im Reservoir. Läuft eine Turbine trocken, wird ihr Leitwerk geschlossen und für eine Fehlerzeit (FT) von 2min gesperrt. Während dieser Zeit kann kein Strom produziert werden.

Ein Überlaufen des Stauwehrs Obertöss bleibt folgenlos; der Überfall wird in das Flussbett der Töss geleitet.
Eine kontrollierte Entlastung des Kanals Niedertöss wird durch Streichwehr gewährt. Da sich dieses im Wohngebiet befindet, beschweren sich Anwohner über das Rauschen. Diese Situation führt in der Environment zu einer Strafzahlung, welche vom Stromerlös abgezogen wird.

## Reinforcement Learning

Die Env kann mit einem RL-Agenten aus dem [Stable Baselines3 Zoo](https://github.com/DLR-RM/stable-baselines3) trainiert werden. Ein Beispiel ist in den Dateien `train_PPO.py` und `test_PPO.py` enthalten.
