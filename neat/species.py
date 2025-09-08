import random


class Species:
    def __init__(self, mem):
        self.members = []
        self.members.append(mem)
        self.rep = self.members[0]
        self.max_members = 8
        self.threshold = 3.5

        self.average_fitness = 0
        self.allowed_offspring = 0

        self.staleness = 0
        pass

    def add(self, brain):
        self.members.append(brain)
        # TODO: Check fitness and set as rep
        if self.rep.fitness < brain.fitness:
            self.rep = self.members[-1]
        pass

    def get_random_parent(self):
        if not self.members:
            return None
        
        if len(self.members) == 1:
            return self.members[0]

        fitness_values = [max(0.1, m.fitness) for m in self.members]
        total_fitness = sum(fitness_values)
        
        if total_fitness <= 0:
            return random.choice(self.members)
        
        selection = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(fitness_values):
            current += fitness
            if current >= selection:
                return self.members[i]

        return self.members[-1]

    def give_offspring(self, mutation_intensity=1.0):
        """Generate offspring with adaptive mutation intensity"""
        parent1 = self.get_random_parent()
        parent2 = self.get_random_parent()
        child = parent1.crossover(parent2)
        child.mutate(mutation_intensity)
        return child

    def check(self, brain):
        done = False
        cd = self.rep.calculate_compatibility(brain)
        if cd < self.threshold and len(self.members) < self.max_members:
            done = True
        return done

    def adjust_fitness(self):
        """Better fitness sharing"""
        if not self.members:
            return
        
        for member in self.members:
            member.adjusted_fitness = member.fitness / len(self.members)
        pass

    def get_average_fitness(self):
        self.average_fitness = sum([(g.adjusted_fitness) for g in self.members]) / len(
            self.members
        )
        return self.average_fitness

    def update_staleness(self, current_best_fitness):
        """Update how long this species has been stagnant"""
        if current_best_fitness > self.rep.fitness:
            self.staleness = 0
            self.rep.fitness = current_best_fitness
        else:
            self.staleness += 1