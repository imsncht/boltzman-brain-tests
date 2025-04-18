import pygame
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

# Config
WIDTH, HEIGHT = 800, 600
NUM_PARTICLES = 200
RADIUS = 2
FPS = 60

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boltzmann Brain - Particle Sim")
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

        if self.x <= RADIUS or self.x >= WIDTH - RADIUS:
            self.vx *= -1
        if self.y <= RADIUS or self.y >= HEIGHT - RADIUS:
            self.vy *= -1

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), RADIUS)

# Collision detection
def collide(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist = math.hypot(dx, dy)
    if dist < 2 * RADIUS:
        p1.vx, p2.vx = p2.vx, p1.vx
        p1.vy, p2.vy = p2.vy, p1.vy

# Observer neural net with memory (LSTM)
class ObserverNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        x = x.unsqueeze(0)  # Add batch dimension
        out, self.hidden = self.lstm(x, self.hidden)
        return self.fc(out[:, -1, :])

    def reset_memory(self):
        self.hidden = None

# Initialize particles and observer
particles = [Particle() for _ in range(NUM_PARTICLES)]
observer = ObserverNet(input_size=NUM_PARTICLES * 4, hidden_size=128, output_size=NUM_PARTICLES * 2)
optimizer = optim.Adam(observer.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

prev_state = []
history_length = 5
frame = 0
running = True

# Entropy log
entropy_log = []

while running:
    clock.tick(FPS)
    screen.fill((10, 10, 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move and draw particles
    current_state = []
    for i, p in enumerate(particles):
        p.move()
        for j in range(i + 1, len(particles)):
            collide(p, particles[j])
        p.draw()
        current_state.extend([p.x / WIDTH, p.y / HEIGHT, p.vx, p.vy])

    current_tensor = torch.tensor(current_state, dtype=torch.float32)
    prev_state.append(current_tensor)
    if len(prev_state) > history_length:
        prev_state.pop(0)

    if len(prev_state) == history_length:
        observer.reset_memory()
        input_seq = torch.stack(prev_state).unsqueeze(0)  # (1, seq_len, features)
        predicted_pos = observer(input_seq[0])

        actual_pos = torch.tensor([p.x / WIDTH for p in particles] +
                                  [p.y / HEIGHT for p in particles], dtype=torch.float32)

        loss = loss_fn(predicted_pos, actual_pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Entropy estimate: std deviation of velocities
        entropy = torch.std(torch.tensor([p.vx for p in particles] + [p.vy for p in particles]))
        entropy_log.append(entropy.item())

        if loss.item() < 0.001 or entropy.item() < 0.1:
            print(f"[Awareness Event] Frame {frame} | Loss: {loss.item():.6f} | Entropy: {entropy.item():.4f}")

        pygame.display.set_caption(f"Boltzmann Brain - Frame {frame}, Loss: {loss.item():.6f}, Entropy: {entropy.item():.4f}")

    pygame.display.flip()
    frame += 1

pygame.quit()
