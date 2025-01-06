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

Ziel der Umgebing ist die Maximierung der Stromproduktion, respektive des Ertrags aus ebendieser. Die Environment kann mit dem Befehl `python play.py` gestartet werden.

![<img src="/img/letsplay.jpg" width="20"/>](/img/letsplay.jpg)

### Beschreibung der Anlage

Das Stauwehr Obertöss (OT) bildet ein Wasserreservoir, aus dem die zwei Turbinen (Index 1 und 2) der ersten Kraftwerkstufe gespiesen werden. Der Abfluss aus den Turbinen 1 und 2 strömt in den ein zweites Reservoir aus, den Kanal Niedertöss (NT). Aus diesem Reservoir werden die zwei Turbinen (Index 3 und 4) der zweiten Kraftwerkstufe gespiesen. Nach der zweiten Kraftwerkstufe fliesst die entnommene Wassermenge zurück in die Töss.

Der Durchfluss durch die Turbinen 1-4 kann individuell durch die Stellung des Leitwerkes (Öffnung 0...100%) eingestellt werden. Die Turbinen der jeweiligen Kraftwerkstufe sind parallel angeordnet und können unabhängig voneinander betrieben werden. Ab einer hydraulischen Leistung von rund 30% des Nominalwertes einer Turbine wird elektrische Energie produziert. 

Parallel zu den Turbinen 1 und 2 kann über einen By-Pass (Index 5) Wasser aus dem Stauwehr OT direkt in den Kanal NT geleitet werden. Über einen weiteren By-Pass (Index 6) kann der Kanal NT unter umgehung der Turbinen 3 und 4 entlastet werden.
Der Durchfluss durch den By-Pass 5 und den By-Pass 6 kann durch die jeweilige Schieberstellung (Öffnung 0...100%) eingestellt werden.


### Kleinwasserkraftwerk Töss

<!-- Das betrachtete System besteht aus einer Kaskade von zwei Kraftwerksstufen. Beim Hauptwehr in Obertöss werden bis zu 5m$^3/s$ Wasser entnommen und über eine Fallhöhe von rund 5.5m in zwei parallel angeordneten Kaplan-Turbinen verstromt. Anschliessend wird das Wasser im alten Rieter-Kanal, welcher neben der Töss verläuft, bis zur zweiten Stufe in Niedertöss geleitet und mit weiteren 5m Fallhöhe in zwei Francis-Turbinen verstromt. -->

## Reinforcement Learning


