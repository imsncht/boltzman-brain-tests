import pygame
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

class ObserverNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# Config
WIDTH, HEIGHT = 800, 600
NUM_PARTICLES = 100
RADIUS = 1
FPS = 60

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Particle Collision Simulation")
clock = pygame.time.Clock()

# Particle class
class Particle:
    def __init__(self):
        self.x = random.uniform(RADIUS, WIDTH - RADIUS)
        self.y = random.uniform(RADIUS, HEIGHT - RADIUS)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def move(self):
        self.x += self.vx
        self.y += self.vy

        # Wall collisions
        if self.x <= RADIUS or self.x >= WIDTH - RADIUS:
            self.vx *= -1
        if self.y <= RADIUS or self.y >= HEIGHT - RADIUS:
            self.vy *= -1

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), RADIUS)

# Check for collision between two particles
def collide(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    distance = math.hypot(dx, dy)
    if distance < 2 * RADIUS:
        # Simple elastic collision (swapping velocities)
        p1.vx, p2.vx = p2.vx, p1.vx
        p1.vy, p2.vy = p2.vy, p1.vy

# Init particles
particles = [Particle() for _ in range(NUM_PARTICLES)]

# Main loop
running = True
while running:
    clock.tick(FPS)
    screen.fill((10, 10, 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update and draw particles
    for i, p in enumerate(particles):
        p.move()
        for j in range(i + 1, len(particles)):
            collide(p, particles[j])
        p.draw()

    # Setup observer
observer = ObserverNet(input_size=NUM_PARTICLES * 4, hidden_size=128, output_size=NUM_PARTICLES * 2)
optimizer = optim.Adam(observer.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Previous state buffer
prev_state = None

for frame in range(100000):  # Infinite simulation
    # Particle simulation code...
    
    # Step 1: Get current state
    current_state = []
    for p in particles:
        current_state.extend([p.x / WIDTH, p.y / HEIGHT, p.vx, p.vy])  # Normalize position

    current_state = torch.tensor(current_state, dtype=torch.float32)

    # Step 2: Train observer to predict positions based on previous state
    if prev_state is not None:
        predicted_pos = observer(prev_state)
        actual_pos = torch.tensor([p.x / WIDTH for p in particles] +
                                  [p.y / HEIGHT for p in particles], dtype=torch.float32)

        loss = loss_fn(predicted_pos, actual_pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Frame {frame}] Observer loss: {loss.item():.6f}")

    prev_state = current_state.clone().detach()

    pygame.display.flip()

pygame.quit()


