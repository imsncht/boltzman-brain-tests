import random
import pygame
from config import WIDTH, HEIGHT, PARTICLE_COUNT, WHITE

class EntropyField:
    def __init__(self):
        self.particles = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(PARTICLE_COUNT)]

    def update(self):
        self.particles = [(x + random.randint(-1, 1), y + random.randint(-1, 1)) for x, y in self.particles]

    def draw(self, screen):
        for x, y in self.particles:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                screen.set_at((x, y), WHITE)
