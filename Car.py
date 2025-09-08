import pygame
import os
import math
import sys
import random

from neat.genome import Genome
from neat.population import Population

screen_width = 1500
screen_height = 800


CHECKPOINTS = [
    [700, 650],  # Start position
    [1185, 673],
    [1270, 550],
    [1130, 85],
    [185, 190],
    [1058, 350],
    [854, 522],
    [540, 207],
    [207, 540],
    [536, 695],
    # [700, 650]  # Back to start for lap completion
]

class Car:
    def __init__(self, gh, clone=False):
        self.surface = pygame.image.load("mcl.png")
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = [700, 650]
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.four_points = []
        self.is_alive = True

        self.distance = 0
        self.time_spent = 0
        self.fitness = 0
        self.gh = gh

        self.last_checkpoint = -1
        self.laps_completed = 0

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
        screen.blit(self.rotate_surface, self.pos)
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

        self.speed = 15
        self.rotate_surface = self.rot_center(self.surface, self.angle)

        # Move car
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        self.distance += self.speed
        self.time_spent += 1

        # Clamp
        self.pos[0] = max(20, min(self.pos[0], screen_width - 120))
        self.pos[1] = max(20, min(self.pos[1], screen_height - 120))

        # Collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        length = 40
        self.four_points = [
            [self.center[0] + math.cos(math.radians(360 - (self.angle + deg))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle + deg))) * length]
            for deg in (30, 150, 210, 330)
        ]

        self.check_collision(map_img)

        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map_img)

        inputs = self.get_inputs()
        outs = self.brain.get_outputs(inputs)
        if outs[0] > outs[1]:
            self.angle += 10
        else:
            self.angle -= 10

        self.update_checkpoint()

        # Update fitness
        # Base fitness: distance traveled + bonus for checkpoints
        checkpoint_bonus = (self.last_checkpoint + 1) * 1000
        lap_bonus = self.laps_completed * len(CHECKPOINTS) * 1500
        self.fitness = self.distance / 50.0 + checkpoint_bonus + lap_bonus

    def update_checkpoint(self):
        """Update last checkpoint hit if car is close enough to it, and track laps"""
        for i, cp in enumerate(CHECKPOINTS):
            cp_x, cp_y = cp
            dist = math.hypot(self.center[0] - cp_x, self.center[1] - cp_y)
            if dist < 50:
                if i == 0 and self.last_checkpoint == len(CHECKPOINTS) - 1:
                    self.laps_completed += 1
                    self.last_checkpoint = -1
                elif i > self.last_checkpoint:
                    self.last_checkpoint = i
                break

    def get_inputs(self):
        """Return normalized radar distances as inputs"""
        inputs = [0, 0, 0, 0, 0]
        for i, r in enumerate(self.radars):
            inputs[i] = r[1] / 300.0
        return inputs

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


def run_car(pop, generation, max_steps=1000):
    cars = [Car(pop.gh) for _ in range(pop.pop_len)]

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
        text = font.render(f"Remain cars : {remain}", True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=(screen_width / 2, 200)))

        pygame.display.flip()
        clock.tick(0)
        steps += 1

    # Copy fitness back into population
    for i, car in enumerate(cars):
        pop.population[i].fitness = car.fitness


if __name__ == "__main__":
    pop = Population(100, 5, 2)  # same as Bird: 5 inputs, 2 outputs
    generations = 100

    for gen in range(generations):
        run_car(pop, gen)
        pop.reset()

        best = max(pop.population, key=lambda x: x.fitness)
        print(f"Gen {gen} | Best fitness: {best.fitness:.2f}")
