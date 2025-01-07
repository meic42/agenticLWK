import gymnasium as gym
import numpy as np
import pygame
from pygame.locals import *


class LWKToess(gym.Env):
    """ 
    ## Laufwasserkraftwerk Toess
    Gymnasium Environment zum Trainieren von RL-Agenten.
    
    ## Hydraulisches Modell

    ## Beschreibung der Komponenten
    T1: Turbine 1 in Obertöss, 170kW
    T2: Turbine 2 in Obertöss, 100kW
    T3: Turbine 3 in Niedertöss, 150kW
    T4: Turbine 4 in Niedertöss, 90kW
    BP5: By-Pass in Obertöss
    BP6: By-Pass in Niedertöss

    Die Komponenten in Obertöss operieren augrund dem Niveau h_OT, 
    die Komponenten in Niedertöss zwischen dem Niveau h_NT.

    Lösungsvariablen 
    h_OT
    h_K1
    h_K2
    h_NT
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, sim_time=600):
        super(LWKToess, self).__init__()

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.elapsed_time = None
        self.sim_time = sim_time

        self.action_space = gym.spaces.Discrete(13)  # Schieber und Leitwerke steuern
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10, 10, 200, 200, 200, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200]),
            dtype=np.float32
        )

        # Kopie der Klassenvariable für die Instanz erstellen
        self.metadata = dict(self.__class__.metadata)  # Kopiert die Klassenvariable

        # Ermittlung des Zeitinkrementes dtime. Ohne Rendering aus Performancegründen wird dtime Grösser gewählt,
        # jedoch nur so gross, dass mit der finiten Änderungsgeschwindigkeit eine hinreichende Einstellgenauigkeit erreicht wird.

        if self.render_mode == None:
            self.metadata["render_fps"] = 5

        self.dtime = 1.0 / self.metadata["render_fps"]

        # Konstanten
        self.g = 10  # Gravitationsbeschleunigung
        self.rho = 1000  # Dichte des Wassers

        # Parameter
        self.h_OT_max = 0.1     # m
        self.h_NT_max = 0.1     # m
        self.hf_OT_0 = 5.5      # m
        self.hf_NT_0 = 5        # m
        self.c_OT = 4000        # m3/m
        self.c_NT = 3200        # m3/m
        self.A1_max = 0.455     # m2
        self.A2_max = 0.27      # m2
        self.A3_max = 0.20      # m2
        self.A4_max = 0.40      # m2
        self.A5_max = 0.70      # m2
        self.A6_max = 0.70      # m2
        self.alpha_speed = 10   # % pro Sekunde
        self.Q_Rest = 0.5       # m3/s
        self.FT_par = 120       # s

        # Deklaration der Lösungsvariablen
        self.h_OT = None        # m
        self.h_NT = None        # m
        self.alpha_1 = None     # %
        self.alpha_2 = None     # %
        self.alpha_3 = None     # %
        self.alpha_4 = None     # %
        self.alpha_5 = None     # %
        self.alpha_6 = None     # %
        self.Pel1 = None        # kW
        self.Pel2 = None        # kW
        self.Pel34 = None       # kW
        self.FT1 = None         # min, verbleibende Sperrzeit Turbine 1
        self.FT2 = None         # min, verbleibende Sperrzeit Turbine 2
        self.FT3 = None         # min, verbleibende Sperrzeit Turbine 3
        self.FT4 = None         # min, verbleibende Sperrzeit Turbine 4
        self.FS_NT = None       # Failure State Ueberfall Streichwehr Niedertoess 
        self.cumulated_reward = None

    def step(self, action):
        # Validierung der Aktion
        assert 0 <= action <= 12, "Invalid action provided!"
        
        # Aktionen auf Schieber und Leitwerke anwenden
        # Bei einzelnen Aktionen wird die Zulässigkeit geprüft über die Fehler-Toleranzzeit (FT)
        dalpha = self.dtime * self.alpha_speed
        if action == 1:
            if not self.FT1 > 0:
                self.alpha_1 = min(self.alpha_1 + dalpha, 100)
        elif action == 2:
            if not self.FT1 > 0:
                self.alpha_1 = max(self.alpha_1 - dalpha, 0)
        elif action == 3:
            if not self.FT2 > 0:
                self.alpha_2 = min(self.alpha_2 + dalpha, 100)
        elif action == 4:
            if not self.FT2 > 0:
                self.alpha_2 = max(self.alpha_2 - dalpha, 0)
        elif action == 5:
            if True:
                self.alpha_5 = min(self.alpha_5 + dalpha, 100)
        elif action == 6:
            if True:
                self.alpha_5 = max(self.alpha_5 - dalpha, 0)
        elif action == 7:
            if not self.FT3 > 0:
                self.alpha_3 = min(self.alpha_3 + dalpha, 100)
        elif action == 8:
            if not self.FT3 > 0:
                self.alpha_3 = max(self.alpha_3 - dalpha, 0)
        elif action == 9:
            if not self.FT4 > 0:
                self.alpha_4 = min(self.alpha_4 + dalpha, 100)
        elif action == 10:
            if not self.FT4 > 0:
                self.alpha_4 = max(self.alpha_4 - dalpha, 0)
        elif action == 11:
            if True:
                self.alpha_6 = min(self.alpha_6 + dalpha, 100)
        elif action == 12:
            if True:
                self.alpha_6 = max(self.alpha_6 - dalpha, 0)
        elif action == 0:
            pass    # keine Aktion
        
        # Fehlerzustaende, Fehlerzeiten dekrementieren (falls > 0)
        self.FT1 = max(0, self.FT1 - self.dtime)
        self.FT2 = max(0, self.FT2 - self.dtime)
        self.FT3 = max(0, self.FT3 - self.dtime)
        self.FT4 = max(0, self.FT4 - self.dtime)

        # OBERTÖSS
        # Durchfluss durch die einzelnen Komponenten anhand der Fallhöhe
        u125 = np.sqrt(2 * self.g * (self.h_OT+self.hf_OT_0))

        Q1 = u125 * self.A1_max * (self.alpha_1 / 100)
        Q2 = u125 * self.A2_max * (self.alpha_2 / 100)
        Q5 = u125 * self.A5_max * (self.alpha_5 / 100)

        # Netto-Zufluss in Obertöss unter Berücksichtigung der gesetzlichen Restwassermenge
        Q_0 = 8.5 # m3/s
        Q_OT_net = Q_0 - self.Q_Rest

        # Maximal mögliche Gesamtentnahme in diesem Zeitschritt damit h_OT nicht negativ wird
        Q125_max = self.h_OT*self.c_OT/self.dtime + Q_OT_net
        
        # Prüfung ob die Gesamtentnahme überschritten wurde
        if (Q1 + Q2 + Q5) > Q125_max:
            if self.alpha_1 > 0:
                self.alpha_1, Q1 = 0, 0 # Störung
                self.FT1 = self.FT_par
            if self.alpha_2 > 0:
                self.alpha_2, Q2 = 0, 0 # Störung
                self.FT2 = self.FT_par
            Q5 = min(Q5, Q125_max) # Q5 begrenzt, so dass h_OT >= 0 bleibt

        # Wasserstandsaktualisierung
        self.h_OT = self.h_OT + (Q_OT_net - Q1 - Q2 - Q5) / self.c_OT * self.dtime
        self.h_OT = np.clip(self.h_OT,0,self.h_OT_max)

        # NIEDERTÖSS

        # Durchfluss durch die einzelnen Komponenten anhand der Fallhöhe
        u346 = np.sqrt(2 * self.g * (self.h_NT+self.hf_NT_0))

        Q3 = u346 * self.A3_max * (self.alpha_3 / 100)
        Q4 = u346 * self.A4_max * (self.alpha_4 / 100)
        Q6 = u346 * self.A6_max * (self.alpha_6 / 100)

        # Zufluss in Kanal
        Q_NT_net = Q1 + Q2 + Q5

        # Maximal mögliche Gesamtentnahme in diesem Zeitschritt damit h_NT nicht negativ wird
        Q346_max = self.h_NT*self.c_NT/self.dtime + Q_NT_net

        # Prüfung ob die Gesamtentnahme überschritten wurde
        if (Q3 + Q4 + Q6) > Q346_max:
            if self.alpha_3 > 0:
                self.alpha_3, Q3 = 0, 0 # Störung
                self.FT3 = self.FT_par
            if self.alpha_4 > 0:
                self.alpha_4, Q4 = 0, 0 # Störung
                self.FT3 = self.FT_par
            Q6 = min(Q6, Q346_max) # Q6 begrenzt, so dass h_NT >= 0 bleibt

        # Wasserstandsaktualisierung
        self.h_NT = self.h_NT + (Q_NT_net - Q3 - Q4 - Q6) / self.c_NT * self.dtime
        
        # Fehlerstatus Streichwehr; Flipflop wenn Füllstand wieder unter max gefallen ist
        if self.h_NT > self.h_NT_max and self.FS_NT == False:
            self.FS_NT = True
            laermklage = True       
        elif self.h_NT < self.h_NT_max and self.FS_NT == True:
            self.FS_NT = False
            laermklage = False
        else:
            laermklage = False

        self.h_NT = np.clip(self.h_NT,0,self.h_NT_max)

        # TURBINEN
        # hydraulische Leistung
        Ph1 = 0.5 * self.rho * u125**2 * Q1 / 1000
        Ph2 = 0.5 * self.rho * u125**2 * Q2 / 1000
        Ph3 = 0.5 * self.rho * u346**2 * Q3 / 1000
        Ph4 = 0.5 * self.rho * u346**2 * Q4 / 1000

        # elektrische Leistung
        self.Pel1 = self._get_Pel(Ph1, 210)
        self.Pel2 = self._get_Pel(Ph2, 115)
        self.Pel34 = self._get_Pel(Ph3+Ph4, 225)

        # Reward
        r_kWh = 0.3 # CHF/kWh
        r_laermklage = -5 # CHF/laermklage
        reward = (self.Pel1 + self.Pel2 + self.Pel34) * self.dtime * r_kWh / 3600 # CHF
        if laermklage == True:
            reward += r_laermklage
        self.cumulated_reward += reward

        # Simulationszeit
        self.elapsed_time += self.dtime
        
        # Termination
        if self.elapsed_time >= self.sim_time:
            truncated = True
            terminated = True
        else:
            truncated = False
            terminated = False
        
        # Rückgabe von Informationen
        info = {
            "cumulated_reward": self.cumulated_reward  # Kumulierten Erlös zurückgeben
        }

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=42, options=None):
        self.h_OT = 0.1 * (self.h_OT_max) #np.random.uniform(0, self.h_OT_max)
        self.h_NT = 0.1 * (self.h_NT_max) #np.random.uniform(0, self.h_OT_max)
        self.alpha_1 = 0
        self.alpha_2 = 0
        self.alpha_3 = 0
        self.alpha_4 = 0
        self.alpha_5 = 0
        self.alpha_6 = 0
        self.Pel1 = 0
        self.Pel2 = 0
        self.Pel34 = 0
        self.FT1 = 0
        self.FT2 = 0
        self.FT3 = 0
        self.FT4 = 0
        self.FS_NT = False
        self.cumulated_reward = 0
        self.elapsed_time = 0

        info = {
            "cumulated_reward": self.cumulated_reward  # Kumulierten Erlös zurückgeben
        }

        return self._get_obs(), info

    def _get_obs(self):
        return np.array([self.h_OT, self.h_NT, self.Pel1, self.Pel2, self.Pel34, self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, self.alpha_5, self.alpha_6, self.FT1, self.FT2, self.FT3, self.FT4], dtype=np.float32)

    def _get_Pel(self, Ph, Ph_eta_max):
        """
        Wirkungsgrad eta := Pel/Ph
        Der maximale Wirkungsgrad von 0.80 wird bei einem Ph_eta_max erreicht
        Elektrische Leistung erst ab 30% der maximalen hydraulischen Leistung
        """
        if Ph >= 0.3*Ph_eta_max:
            eta = - (0.81 / Ph_eta_max**2) * (Ph - Ph_eta_max)**2 + 0.81
        else:
            eta = 0
        return Ph * eta

    def render(self):
        
        if self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode '{self.render_mode}'. Valid modes are {self.metadata['render_modes']}.")

        if not hasattr(self, "screen"):
            self._initialize_pygame()
        
        self._clear_screen()
        self._draw_obertoess()
        self._draw_niedertoess()
        self._draw_status()
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return self._get_rgb_array()

    def _initialize_pygame(self):
        """Initialisiert Pygame-Elemente."""
        pygame.init()
        self.window_size = 700
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size), pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Laufwasserkraftwerk Töss")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

    def _clear_screen(self):
        """Löscht den Bildschirm."""
        self.screen.fill((255, 255, 255))  # Weiß

    def _draw_obertoess(self):
        """Zeichnet den Bereich Obertöss."""
        colors = {"water": (0, 0, 255), "tank_outline": (0, 0, 0)}
        h_OT_display = int(self.h_OT / self.h_OT_max * 50)
        
        # Wasser
        pygame.draw.polygon(
            self.screen,
            colors["water"],
            [
                (300, 100),
                (300, 150),
                (150, 150),
                (25, 110),
                (25, 100),
                (25, 100 - h_OT_display),
                (300, 100 - h_OT_display),
            ]
        )
        pygame.draw.line(self.screen, colors["tank_outline"], (25, 100), (300, 100), 1)
        pygame.draw.line(self.screen, colors["tank_outline"], (25, 50), (300, 50), 1)
        
        # Text
        self._draw_text(f"Wasserstand OT: {self.h_OT:.3f} m", (50, 69))
        self._draw_text(f"Turbine 1: Pel {self.Pel1:.0f} kW, Leitwerk {self.alpha_1:.0f} %", (25, 200))
        self._draw_text(f"Fehler-Restzeit T1: {self.FT1:.1f} s", (350, 200))
        self._draw_text(f"Turbine 2: Pel {self.Pel2:.0f} kW, Leitwerk {self.alpha_2:.0f} %", (25, 230))
        self._draw_text(f"Fehler-Restzeit T2: {self.FT2:.1f} s", (350, 230))
        self._draw_text(f"Bypass OT: Schieberöffnung {self.alpha_5:.0f} %", (25, 260))

    def _draw_niedertoess(self):
        """Zeichnet den Bereich Niedertöss."""
        colors = {"water": (0, 0, 255), "tank_outline": (0, 0, 0)}
        h_NT_display = int(self.h_NT / self.h_NT_max * 50)
        
        # Wasser
        pygame.draw.polygon(
            self.screen,
            colors["water"],
            [
                (300, 400),
                (300, 450),
                (150, 450),
                (25, 410),
                (25, 400),
                (25, 400 - h_NT_display),
                (300, 400 - h_NT_display),
            ]
        )
        pygame.draw.line(self.screen, colors["tank_outline"], (25, 400), (300, 400), 1)
        pygame.draw.line(self.screen, colors["tank_outline"], (25, 350), (300, 350), 1)
        
        # Text
        self._draw_text(f"Wasserstand NT: {self.h_NT:.3f} m", (50, 369))
        if self.FS_NT:
            self._draw_text("Überfall! -500 CHF!", (350, 369))
        self._draw_text(f"Turbine 3: Leitwerk {self.alpha_3:.0f} %", (25, 500))
        self._draw_text(f"Fehler-Restzeit T3: {self.FT3:.1f} s", (350, 500))
        self._draw_text(f"Turbine 4: Leitwerk {self.alpha_4:.0f} %", (25, 530))
        self._draw_text(f"Fehler-Restzeit T4: {self.FT4:.1f} s", (350, 530))
        self._draw_text(f"Turbine 3+4: {self.Pel34:.0f} kW", (25, 560))
        self._draw_text(f"Bypass NT: Schieberöffnung {self.alpha_6:.0f} %", (25, 590))

    def _draw_status(self):
        """Zeichnet den Statusbereich."""
        self._draw_text(f"Erlös aus Stromverkauf: {self.cumulated_reward:.1f} CHF", (350, 69))
        self._draw_text(f"Dauer der Simulation: {self.elapsed_time:.1f} s", (350, 99))

    def _draw_text(self, text, position):
        """Hilfsfunktion zum Zeichnen von Text."""
        img = self.font.render(text, False, (0, 0, 0), (255, 255, 255))
        self.screen.blit(img, position)

    def _get_rgb_array(self):
        """Gibt den aktuellen Bildschirm als RGB-Array zurück."""
        np_array = pygame.surfarray.array3d(self.screen)
        return np.transpose(np_array, (1, 0, 2))


    def close(self):
        """Schließt Pygame-Ressourcen, falls vorhanden."""
        if hasattr(self, "screen"):
            pygame.display.quit()
            pygame.quit()



# **************************************************************************
