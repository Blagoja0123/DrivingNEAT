from neat.genome import Genome
from neat.geneh import GeneHistory
from neat.species import Species
import random
import math


class Population:
    def __init__(self, pop_len, n_inputs, n_outputs, elite_count=2):
        self.pop_len = pop_len
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.population = []
        self.gh = GeneHistory(n_inputs, n_outputs)

        for _ in range(pop_len):
            self.population.append(Genome(self.gh))

        self.species = []
        self.global_avg = 0
        self.elites = []
        self.elite_count = elite_count
        
        # TTL tracking
        self.best_fitness_history = []
        self.current_ttl = 0
        self.previous_ttl = 0
        self.max_ttl_without_improvement = 10

    def speciate(self):
        """Group genomes into species based on compatibility"""
        species_assigned = [False for _ in range(len(self.population))]
        self.species.clear()

        while False in species_assigned:
            p = random.randint(0, self.pop_len - 1)
            while species_assigned[p]:
                p = random.randint(0, self.pop_len - 1)

            rep = self.population[p]
            sp = Species(rep)
            species_assigned[p] = True

            for i in range(self.pop_len):
                if i == p or species_assigned[i]:
                    continue
                if sp.check(self.population[i]):
                    sp.add(self.population[i])
                    species_assigned[i] = True

            self.species.append(sp)

    def set_allowed_offspring(self):
        """Calculate how many offspring each species should produce"""
        if not self.species:
            return
            
        total_fitness = 0
        valid_species = []

        for sp in self.species:
            sp.adjust_fitness()
            avg_fitness = sp.get_average_fitness()

            if math.isnan(avg_fitness) or math.isinf(avg_fitness) or avg_fitness <= 0:
                sp.allowed_offspring = 1
                continue
                
            valid_species.append(sp)
            total_fitness += avg_fitness

        if not valid_species or total_fitness <= 0 or math.isnan(total_fitness):
            base_offspring = max(1, (self.pop_len - self.elite_count) // len(self.species))
            for sp in self.species:
                sp.allowed_offspring = base_offspring
            return

        self.global_avg = total_fitness / len(valid_species)
        remaining_offspring = self.pop_len - self.elite_count

        total_assigned = 0
        for sp in valid_species:
            if math.isnan(sp.average_fitness) or math.isnan(self.global_avg) or self.global_avg == 0:
                sp.allowed_offspring = 1
            else:
                proportion = sp.average_fitness / self.global_avg
                offspring_share = proportion * remaining_offspring / len(self.species)

                if math.isnan(offspring_share) or math.isinf(offspring_share):
                    sp.allowed_offspring = 1
                else:
                    sp.allowed_offspring = max(1, round(offspring_share))
            
            total_assigned += sp.allowed_offspring
        
        # Handle any rounding discrepancies
        if total_assigned != remaining_offspring:
            diff = remaining_offspring - total_assigned
            if valid_species:
                largest_species = max(valid_species, key=lambda s: s.allowed_offspring)
                largest_species.allowed_offspring = max(1, largest_species.allowed_offspring + diff)

    def reset(self):
        """Create the next generation with adaptive mutation"""
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        current_best = self.population[0].fitness
        

        self.update_ttl(current_best)

        mutation_intensity = self.calculate_mutation_intensity()

        self.elites = []
        for i in range(self.elite_count):
            elite_clone = self.population[i].clone()
            elite_clone.fitness = 0
            self.elites.append(elite_clone)

        self.speciate()
        self.set_allowed_offspring()

        new_pop = []
        total_offspring_needed = self.pop_len - self.elite_count

        # Generate offspring with adaptive mutation
        for sp in self.species:
            sp.update_staleness(current_best)

            species_mutation_intensity = mutation_intensity
            if sp.staleness > 5:
                species_mutation_intensity *= (1 + sp.staleness * 0.2)
            
            for _ in range(min(sp.allowed_offspring, total_offspring_needed - len(new_pop))):
                child = sp.give_offspring(species_mutation_intensity)
                child.fitness = 0
                new_pop.append(child)

        while len(new_pop) < total_offspring_needed:
            random_genome = Genome(self.gh)
            random_genome.mutate(mutation_intensity * 1.5)
            new_pop.append(random_genome)

        self.population = self.elites + new_pop[:total_offspring_needed]
        
        print(f"Generation reset - Mutation intensity: {mutation_intensity:.2f}, TTL: {self.current_ttl}")

    def update_ttl(self, current_best_fitness):
        """Update time-to-live based on fitness progress"""
        self.best_fitness_history.append(current_best_fitness)
        
        # Keep only recent history
        if len(self.best_fitness_history) > 20:
            self.best_fitness_history.pop(0)
        
        # Check if we've made progress recently
        if len(self.best_fitness_history) >= 5:
            recent_best = max(self.best_fitness_history[-5:])
            older_best = max(self.best_fitness_history[:-5]) if len(self.best_fitness_history) > 5 else 0
            
            if recent_best > older_best * 1.05:
                self.previous_ttl = self.current_ttl
                self.current_ttl = 0
            else:
                self.current_ttl += 1

    def calculate_mutation_intensity(self):
        """Calculate mutation intensity based on stagnation"""
        base_intensity = 1.0

        stagnation_multiplier = 1 + (self.current_ttl / self.max_ttl_without_improvement)

        if self.previous_ttl > 0 and self.current_ttl >= (self.previous_ttl * 0.8):
            stagnation_multiplier *= 1.5
            print(f"Near previous TTL ({self.previous_ttl}), boosting mutations!")

        intensity = min(base_intensity * stagnation_multiplier, 4.0)
        
        return intensity

    def next(self):
        self.best_index = (self.best_index + 1) % self.pop_len
        self.best = self.population[self.best_index]

    def prev(self):
        self.best_index = (self.best_index - 1) % self.pop_len
        self.best = self.population[self.best_index]