import random
import time
import pygame
from config import BRAIN_COLOR, NEURON_COLOR, THOUGHT_COLOR
from datetime import datetime

class BoltzmannBrain:
    def __init__(self, x, y, font):
        self.x = x
        self.y = y
        self.birth_time = time.time()
        self.lifespan = random.randint(4, 7)
        self.thoughts = []
        self.font = font
        self.neurons = self._generate_neurons()
        self.memory = random.sample([
            "A family I never had...",
            "A planet I've never seen...",
            "A melody I can’t forget...",
            "A war that never happened..."
        ], 2)

    def _generate_neurons(self):
        return [(self.x + random.randint(-40, 40), self.y + random.randint(-40, 40)) for _ in range(6)]

    def is_alive(self):
        return (time.time() - self.birth_time) < self.lifespan

    def think(self):
        thought = random.choice([
            "Is this a glitch in reality?",
            "Do I remember... or imagine?",
            "Who watches me think?",
            "Was I born from silence or chaos?"
        ])
        self.thoughts.append((time.time(), thought))
        return thought

    def draw(self, screen):
        # Draw neurons and connections
        for nx, ny in self.neurons:
            pygame.draw.line(screen, NEURON_COLOR, (self.x, self.y), (nx, ny), 1)
            pygame.draw.circle(screen, NEURON_COLOR, (nx, ny), 3)
        # Brain core
        pygame.draw.circle(screen, BRAIN_COLOR, (self.x, self.y), 15)
        # Last thought
        if self.thoughts:
            _, last = self.thoughts[-1]
            label = self.font.render(last, True, THOUGHT_COLOR)
            screen.blit(label, (self.x - label.get_width() // 2, self.y + 25))

    def dissolve(self):
        with open(f"brain_log_{int(self.birth_time)}.txt", "w") as f:
            f.write(f"Boltzmann Brain Log — Created {datetime.fromtimestamp(self.birth_time)}\n")
            f.write("False Memories:\n")
            for mem in self.memory:
                f.write(f" - {mem}\n")
            f.write("\nThoughts:\n")
            for t, thought in self.thoughts:
                f.write(f" [{datetime.fromtimestamp(t).strftime('%H:%M:%S')}] {thought}\n")
