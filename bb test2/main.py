import pygame
import random
import sys
import time
from config import *
from entropy_field import EntropyField
from brain import BoltzmannBrain

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boltzmann Brain Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont('consolas', 18)

field = EntropyField()
brain = None
last_think = 0

while True:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    field.update()
    field.draw(screen)

    # Create brain if rare event occurs
    if not brain and random.random() < RARE_FLUCTUATION_CHANCE:
        bx, by = random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)
        brain = BoltzmannBrain(bx, by, font)
        print("\nBoltzmann Brain formed...")
        for mem in brain.memory:
            print(f"  Memory: {mem}")

    if brain:
        if brain.is_alive():
            if time.time() - last_think > THINK_INTERVAL:
                thought = brain.think()
                print(f"  Thought: {thought}")
                last_think = time.time()
            brain.draw(screen)
        else:
            print("The brain fades back into entropy.\n")
            brain.dissolve()
            brain = None

    pygame.display.flip()
    clock.tick(FPS)
