import random

import numpy as np
import pygame
import os
import math
import sys

from neat.genome import Genome
from neat.population import Population
from vis.LiveNetworkViewer import NetworkVisualizer

screen_width = 1500
screen_height = 800


CHECKPOINTS = [
    [1185, 673],
    [1270, 550],
    [1130, 85],
    [185, 190],
    [575, 265],
    [830, 350],
    [1058, 350],
    [900, 522],
    [207, 540],
    [700, 650],
]

import matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)

fitness_history = {"best": [], "avg": []}
species_history = []
checkpoint_history = []

def plot_checkpoint_visits(checkpoint_visits, generation):
    checkpoints = list(checkpoint_visits.keys())
    counts = list(checkpoint_visits.values())
    plt.bar(checkpoints, counts)
    plt.xlabel("Checkpoint")
    plt.ylabel("Number of Visits")
    plt.title(f"Checkpoint Visits (Gen {generation})")
    plt.tight_layout()
    plt.savefig(f"plots/checkpoint_visits_gen{generation}.png")
    plt.close()

def plot_fitness_history(generation):
    plt.figure(figsize=(10,6))
    plt.plot(fitness_history["best"], label="Best Fitness", color="gold", linewidth=2)
    plt.plot(fitness_history["avg"], label="Average Fitness", color="skyblue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progress Over Generations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/fitness_history_gen{generation}.png")
    plt.close()

def plot_species_diversity(generation):
    plt.figure(figsize=(10,6))
    plt.plot(species_history, color="purple", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Number of Species")
    plt.title("Species Diversity Over Generations")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/species_diversity_gen{generation}.png")
    plt.close()

def plot_checkpoint_heatmap(generation):
    data = np.array(checkpoint_history)  # shape: [generations x checkpoints]
    plt.figure(figsize=(12,6))
    plt.imshow(data.T, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(label="Cars Reaching Checkpoint")
    plt.xlabel("Generation")
    plt.ylabel("Checkpoint Index")
    plt.title("Checkpoint Progress Across Generations")
    plt.tight_layout()
    plt.savefig(f"plots/checkpoint_heatmap_gen{generation}.png")
    plt.close()

def plot_lap_completion(lap_counts, generation):
    plt.figure(figsize=(10,6))
    plt.bar(range(len(lap_counts)), lap_counts, color="limegreen")
    plt.xlabel("Generation")
    plt.ylabel("Laps Completed")
    plt.title("Lap Completion Across Generations")
    plt.tight_layout()
    plt.savefig(f"plots/lap_completion_gen{generation}.png")
    plt.close()

def plot_speed_profile(speed_history, generation):
    plt.figure(figsize=(10,6))
    plt.plot(speed_history, color="red", linewidth=1.5)
    plt.xlabel("Timestep")
    plt.ylabel("Speed")
    plt.title(f"Best Car Speed Profile (Gen {generation})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/speed_profile_gen{generation}.png")
    plt.close()

class Car:
    def __init__(self, gh, clone=False):
        self.rotate_rect = None
        self.surface = pygame.image.load("mcl.png")
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = [700, 650]
        self.angle = 0
        self.speed = 5.0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.four_points = []
        self.is_alive = True

        self.max_speed = 10
        self.min_speed = 5.0
        self.acceleration = 1.0
        self.braking = 4.0

        self.distance = 0
        self.time_spent = 0
        self.fitness = 0
        self.gh = gh

        self.last_checkpoint = -1
        self.laps_completed = 0

        self.speed_history = []
        self.checkpoint_times = []
        self.total_speed_bonus = 0

        self.low_speed_time = 0

        if not clone:
            self.brain = Genome(gh)
            for _ in range(10):
                self.brain.mutate()

    def mate(self, partner):
        child = Car(self.gh)
        child.brain = self.brain.crossover(partner.brain)
        return child

    def clone(self):
        child = Car(self.gh, True)
        child.brain = self.brain.clone()
        return child

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.rotate_rect.topleft)
        for r in self.radars:
            pos, _ = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, map_img):
        self.is_alive = True
        for p in self.four_points:
            if map_img.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map_img):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not map_img.get_at((x, y)) == (255, 255, 255, 255) and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.radars.append([(x, y), dist])

    def update(self, map_img):
        if not self.is_alive:
            return

        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        length = 60
        self.four_points = [
            [self.center[0] + math.cos(math.radians(360 - (self.angle + deg))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle + deg))) * length]
            for deg in (30, 150, 210, 330)
        ]
        self.check_collision(map_img)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map_img)

        if not self.is_alive:
            self.fitness -= 100
            return

        inputs = self.get_inputs()
        inputs.append(self.speed / self.max_speed)
        outs = self.brain.get_outputs(inputs)

        steering = (outs[0] - 0.5) * 2
        accel_output = outs[1]
        brake_output = outs[2]


        if accel_output > 0.3:
            self.speed += self.acceleration
        elif brake_output > 0.7:
            self.speed -= self.braking
        else:

            if self.speed < self.max_speed * 0.8:
                self.speed += self.acceleration * 0.4
            else:
                self.speed += self.acceleration * 0.1

        self.speed = max(self.min_speed, min(self.speed, self.max_speed))
        speed_ratio = (self.speed - self.min_speed) / (self.max_speed - self.min_speed + 1e-6)
        turn_strength = 1.8 - speed_ratio * 1.3

        self.angle += steering * 7 * turn_strength

        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        self.distance += self.speed
        self.time_spent += 1

        self.speed_history.append(self.speed)
        if len(self.speed_history) > 100:
            self.speed_history.pop(0)

        self.pos[0] = max(20, min(self.pos[0], screen_width - 120))
        self.pos[1] = max(20, min(self.pos[1], screen_height - 120))

        self.rotate_surface, self.rotate_rect = self.rot_center(self.surface, self.angle)
        self.rotate_rect.center = (self.center[0], self.center[1])

        self.update_checkpoint_progress()

        speed_ratio = self.speed / self.max_speed

        if speed_ratio > 0.8:
            speed_bonus = speed_ratio * 2.0
        elif speed_ratio > 0.6:
            speed_bonus = speed_ratio * 1.0
        elif speed_ratio > 0.4:
            speed_bonus = speed_ratio * 0.3
        else:
            speed_bonus = -0.5

        self.fitness += speed_bonus
        self.total_speed_bonus += speed_bonus

        if self.speed <= self.min_speed * 1.2:
            self.low_speed_time += 1
            penalty = 0.1 * (self.low_speed_time / 10) ** 2
            self.fitness -= penalty
        else:
            self.low_speed_time = max(0, self.low_speed_time - 2)

        if self.low_speed_time > 200:
            self.is_alive = False
            self.fitness -= 300
            return

        if len(self.speed_history) > 20:
            avg_recent_speed = sum(self.speed_history[-20:]) / 20
            avg_speed_ratio = avg_recent_speed / self.max_speed
            if avg_speed_ratio > 0.7:
                self.fitness += avg_speed_ratio * 1.0
            elif avg_speed_ratio < 0.4:
                self.fitness -= (0.4 - avg_speed_ratio) * 0.8

        if self.time_spent > 0 and speed_ratio > 0.5:
            efficiency = self.distance / self.time_spent
            self.fitness += efficiency * 0.15

        if self.time_spent > 8000:
            self.is_alive = False
            self.fitness -= 200

    def update_checkpoint_progress(self):
        next_cp = (self.last_checkpoint + 1) % len(CHECKPOINTS)
        cp_x, cp_y = CHECKPOINTS[next_cp]
        dist_to_cp = math.hypot(self.center[0] - cp_x, self.center[1] - cp_y)

        if not hasattr(self, "best_dist_to_next_cp"):
            self.best_dist_to_next_cp = dist_to_cp

        if dist_to_cp < self.best_dist_to_next_cp:
            progress = self.best_dist_to_next_cp - dist_to_cp
            self.fitness += progress * 0.1
            self.best_dist_to_next_cp = dist_to_cp

        if dist_to_cp < 60:
            if next_cp != self.last_checkpoint:

                base_reward = 10 + (self.last_checkpoint + 1) * 5
                self.fitness += base_reward
                # print(base_reward)
                # if hasattr(self, 'checkpoint_times') and len(self.checkpoint_times) > 0:
                #     time_since_last = self.time_spent - (self.checkpoint_times[-1] if self.checkpoint_times else 0)
                #     speed_multiplier = max(1.0, 500 / max(time_since_last, 50))
                #     speed_bonus = base_reward * (speed_multiplier - 1.0) * 0.2
                #
                #     total_reward = base_reward + speed_bonus
                #     print(f"Checkpoint {next_cp} reached in {time_since_last} steps! Base: {base_reward:.0f}, Speed bonus: {speed_bonus:.0f}, Total: {total_reward:.0f}")
                # else:
                #     total_reward = base_reward
                #     print(f"Checkpoint {next_cp} reached! Reward: {total_reward}")
                #
                # self.fitness += total_reward
                #
                # self.checkpoint_times.append(self.time_spent)
                #
                # if len(self.speed_history) > 5:
                #     avg_speed_at_checkpoint = sum(self.speed_history[-5:]) / 5
                #     speed_ratio = avg_speed_at_checkpoint / self.max_speed
                #     if speed_ratio > 0.7:
                #         high_speed_bonus = 10 * speed_ratio
                #         self.fitness += high_speed_bonus
                #         print(f"  High-speed checkpoint bonus: {high_speed_bonus:.0f}")

                checkpoint_visits[next_cp] += 1
                print(f"Checkpoint {next_cp} visited {checkpoint_visits[next_cp]} times")
                self.last_checkpoint = next_cp


    def get_inputs(self):
        """Return normalized radar distances as inputs"""
        inputs = [0, 0, 0, 0, 0]
        for i, r in enumerate(self.radars):
            inputs[i] = r[1] / 300.0
        return inputs

    def rot_center(self, image, angle):
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=(self.center[0], self.center[1]))
        return rot_image, rot_rect


def run_car(pop, generation, max_steps=20000):

    cars = []
    for i in range(pop.pop_len):
        car = Car(pop.gh, clone=True)
        car.brain = pop.population[i].clone()
        car.brain.fitness = 0
        cars.append(car)

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    map_img = pygame.image.load('map.png')

    steps = 0
    while steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        remain = 0
        for car in cars:
            if car.is_alive:
                car.update(map_img)
                remain += 1

        if remain == 0:
            break

        # Draw
        screen.blit(map_img, (0, 0))
        for car in cars:
            if car.is_alive:
                car.draw(screen)

        text = generation_font.render(f"Generation : {generation}", True, (255, 255, 0))
        screen.blit(text, text.get_rect(center=(screen_width / 2, 100)))
        text = font.render(f"Remaining cars : {remain}", True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=(screen_width / 2, 200)))

        # Show speed statistics for best performing car
        if cars:
            best_car = max(cars, key=lambda c: c.fitness if c.is_alive else 0)
            if best_car.is_alive and len(best_car.speed_history) > 0:
                current_speed = best_car.speed
                avg_speed = sum(best_car.speed_history) / len(best_car.speed_history)
                speed_text = font.render(f"Best Car - Speed: {current_speed:.1f} | Avg: {avg_speed:.1f} | Speed Bonus: {best_car.total_speed_bonus:.0f}",
                                       True, (0, 255, 0))
                screen.blit(speed_text, speed_text.get_rect(center=(screen_width / 2, 230)))

        pygame.display.flip()
        clock.tick(0)
        steps += 1

    for i, car in enumerate(cars):
        pop.population[i].fitness = car.fitness

    best_genome = max(pop.population, key=lambda x: x.fitness)
    visualizer.visualize_network(best_genome, f"Generation {generation} | Fitness {best_genome.fitness:.2f}")


if __name__ == "__main__":
    random.seed(0)
    pop = Population(250, 6, 3)
    generations = 400
    checkpoint_visits = {i: 0 for i in range(len(CHECKPOINTS))}
    best_ever_fitness = 0
    generation_without_improvement = 0

    visualizer = NetworkVisualizer()

    for gen in range(generations):
        run_car(pop, gen)

        best = max(pop.population, key=lambda x: x.fitness)
        avg_fitness = sum(g.fitness for g in pop.population) / len(pop.population)

        fitness_history["best"].append(best.fitness)
        fitness_history["avg"].append(avg_fitness)
        species_history.append(len(pop.species))
        checkpoint_history.append(list(checkpoint_visits.values()))


        if best.fitness > best_ever_fitness:
            best_ever_fitness = best.fitness
            generation_without_improvement = 0
        else:
            generation_without_improvement += 1

        print(f"Gen {gen:3d} | Best: {best.fitness:8.2f} | Avg: {avg_fitness:6.2f} | "
              f"TTL: {pop.current_ttl} | Stagnant: {generation_without_improvement}")

        pop.reset()

        print(f"         | Species: {len(pop.species)} | Mutation intensity calculated")

        if generation_without_improvement > 25:
            print("Injecting high diversity!")
            for i in range(pop.pop_len // 3):
                new_genome = Genome(pop.gh)
                new_genome.mutate(5.0)
                pop.population[-(i+1)] = new_genome
            generation_without_improvement = 0

        plot_checkpoint_visits(checkpoint_visits, gen)
        plot_fitness_history(gen)
        plot_species_diversity(gen)
        plot_checkpoint_heatmap(gen)

