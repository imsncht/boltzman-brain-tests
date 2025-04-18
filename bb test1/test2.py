import pygame
import random
import sys
import time

# === CONFIG ===
WIDTH, HEIGHT = 800, 600
BRAIN_LIFESPAN = 20  # seconds
PARTICLE_COUNT = 1000

# === COLORS ===
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BRAIN_COLOR = (0, 255, 255)
THOUGHT_COLOR = (255, 0, 255)

# === INIT ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boltzmann Brain Simulator")
clock = pygame.time.Clock()

font = pygame.font.SysFont('consolas', 18)

class EntropyField:
    def __init__(self):
        self.particles = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(PARTICLE_COUNT)]

    def update(self):
        # Chaotic motion
        self.particles = [(x + random.randint(-1, 1), y + random.randint(-1, 1)) for x, y in self.particles]

    def draw(self):
        for x, y in self.particles:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                screen.set_at((x, y), WHITE)

class BoltzmannBrain:
    def __init__(self):
        self.x = random.randint(100, WIDTH - 100)
        self.y = random.randint(100, HEIGHT - 100)
        self.birth_time = time.time()
        self.thoughts = []

    def is_alive(self):
        return (time.time() - self.birth_time) < BRAIN_LIFESPAN

    def think(self):
        thought = random.choice([
            "Am I real?", "Why do I exist?", "Is this a dream?",
            "Will I fade?", "I remember... nothing.", "Do I think, therefore I am?"
        ])
        self.thoughts.append(thought)
        return thought

    def draw(self):
        # Pulse effect
        radius = 30 + int(5 * abs(time.time() % 1 - 0.5))
        pygame.draw.circle(screen, BRAIN_COLOR, (self.x, self.y), radius, 2)
        if self.thoughts:
            label = font.render(self.thoughts[-1], True, THOUGHT_COLOR)
            screen.blit(label, (self.x - label.get_width() // 2, self.y + radius + 10))

# === MAIN LOOP ===
field = EntropyField()
brain = None
last_think_time = 0
THINK_INTERVAL = 1.0  # seconds

while True:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    field.update()
    field.draw()

    # Occasionally form a Boltzmann brain
    if brain is None and random.random() < 0.0005:  # rare event
        brain = BoltzmannBrain()
        print("\n[🧠] A Boltzmann Brain has emerged...")

    # Brain exists
    if brain:
        if brain.is_alive():
            current_time = time.time()
            if current_time - last_think_time > THINK_INTERVAL:
                thought = brain.think()
                print(f"[🧠] {thought}")
                last_think_time = current_time
            brain.draw()
        else:
            print("[🧠] The brain has dissolved into entropy.")
            brain = None

    pygame.display.flip()
    clock.tick(600)
