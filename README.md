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

Ziel des Übung ist die Maximierung der Stromproduktion, respektive des Ertrags aus ebendieser. Die Environment kann mit dem Befehl `python play.py` gestartet werden:

![Screenshot der Environment im Spielbetrieb](/img/letsplay.jpg)

### Spielmodus

### Kleinwasserkraftwerk Töss

Das betrachtete System besteht aus einer Kaskade von zwei Kraftwerksstufen. Beim Hauptwehr in Obertöss werden bis zu 5m$^3/s$ Wasser entnommen und über eine Fallhöhe von rund 5.5m in zwei parallel angeordneten Kaplan-Turbinen verstromt. Anschliessend wird das Wasser im alten Rieter-Kanal, welcher neben der Töss verläuft, bis zur zweiten Stufe in Niedertöss geleitet und mit weiteren 5m Fallhöhe in zwei Francis-Turbinen verstromt.

## Reinforcement Learning


