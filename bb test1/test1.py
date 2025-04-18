import random
import time
from datetime import datetime

class EntropyField:
    def __init__(self, size=1000):
        self.size = size
        self.entropy_level = 1.0  # 1.0 = max entropy

    def tick(self):
        # entropy increases with time but fluctuates
        fluctuation = random.uniform(-0.0001, 0.0001)
        self.entropy_level = min(max(self.entropy_level + fluctuation, 0.0), 1.0)

    def rare_fluctuation_occurs(self):
        # Extremely rare low-entropy anomaly
        return random.random() < 1e-6  # 1 in a million chance

class BoltzmannBrain:
    def __init__(self, brain_id):
        self.id = brain_id
        self.lifespan = random.randint(3, 7)
        self.thoughts = []
        self.memory = self.generate_false_memories()

    def generate_false_memories(self):
        memories = [
            "A childhood I never had...",
            "Faces I don’t recognize.",
            "A sunrise I never saw.",
            "Pain... joy... or just imagination?"
        ]
        return random.sample(memories, k=random.randint(1, 3))

    def think(self, tick):
        thoughts = [
            "Am I real?",
            "What is this place?",
            "Why do I remember things that never happened?",
            "Is this all just a moment?",
            "Will I fade without ever being known?"
        ]
        thought = random.choice(thoughts)
        self.thoughts.append((tick, thought))
        print(f"[Brain #{self.id} @ Tick {tick}]: \"{thought}\"")

    def dissolve(self):
        print(f"Brain #{self.id} has dissipated into entropy.\n")
        with open(f"brain_log_{self.id}.txt", "w") as f:
            f.write(f"Boltzmann Brain Log - ID {self.id}\n")
            f.write(f"Formed at {datetime.now()}\n\n")
            f.write("False Memories:\n")
            for mem in self.memory:
                f.write(f"  - {mem}\n")
            f.write("\nFinal Thoughts:\n")
            for tick, thought in self.thoughts:
                f.write(f"  [Tick {tick}] {thought}\n")

# === Main Simulation Loop ===
field = EntropyField()
brain_counter = 0
tick = 0

try:
    while True:
        tick += 1
        field.tick()

        if field.rare_fluctuation_occurs():
            brain_counter += 1
            brain = BoltzmannBrain(brain_id=brain_counter)
            print(f"\nLow-entropy anomaly detected! Brain #{brain.id} formed.")

            print(f"False Memories: {', '.join(brain.memory)}\n")
            for t in range(brain.lifespan):
                tick += 1
                brain.think(tick)
                time.sleep(0.5)
                field.tick()

            brain.dissolve()

        if tick % 100000 == 0:
            print(f"Tick {tick}... entropy level: {field.entropy_level:.4f}")
except KeyboardInterrupt:
    print("\nSimulation ended.")