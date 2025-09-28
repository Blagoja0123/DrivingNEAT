from neat.node import Node
from neat.gene import Gene

import random


class Genome:
    def __init__(self, gh, c1=1.0, c2=1.0, c3=1.0):
        self.gh = gh
        self.n_inputs = gh.n_inputs
        self.n_outputs = gh.n_outputs
        self.total_nodes = 0

        self.nodes = []
        self.genes = []

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.fitness = 0
        self.adjusted_fitness = 0

        for _ in range(self.n_inputs):
            self.nodes.append(Node(self.total_nodes, 0))
            self.total_nodes += 1

        for _ in range(self.n_outputs):
            self.nodes.append(Node(self.total_nodes, 1))
            self.total_nodes += 1

        self.create_initial_connections()

    def create_initial_connections(self):
        """Create initial fully connected network from inputs to outputs"""
        for i in range(self.n_inputs):
            for o in range(self.n_outputs):
                input_node = self.nodes[i]
                output_node = self.nodes[self.n_inputs + o]
                self.connect_nodes(input_node, output_node)

    def clone(self):
        clone = Genome(self.gh)
        clone.total_nodes = self.total_nodes
        clone.nodes.clear()
        clone.genes.clear()

        for i in range(len(self.nodes)):
            clone.nodes.append(self.nodes[i].clone())

        for i in range(len(self.genes)):
            clone.genes.append(self.genes[i].clone())

        clone.connect_genes()
        return clone

    def exists(self, inno):
        for g in self.genes:
            if g.inno == inno:
                return True
        return False

    def connect_nodes(self, n1, n2):
        n1layer = n1.layer if n1.layer != 1 else 1000000
        n2layer = n2.layer if n2.layer != 1 else 1000000

        if n1layer > n2layer:
            n1, n2 = n2, n1

        c = self.gh.exists(n1, n2)
        x = Gene(n1, n2)

        if c:
            x.inno = c.inno
            if not self.exists(x.inno):
                self.genes.append(x)
        else:
            x.inno = self.gh.global_inno
            self.gh.global_inno += 1
            self.gh.all_genes.append(x.clone())
            self.genes.append(x)
        pass

    def add_gene(self):
        n1 = random.choice(self.nodes)
        n2 = random.choice(self.nodes)

        while n1.layer == n2.layer:
            n1 = random.choice(self.nodes)
            n2 = random.choice(self.nodes)

        self.connect_nodes(n1, n2)
        pass

    def mutate(self, mutation_intensity=1.0):
        if len(self.genes) == 0:
            self.add_gene()

        weight_mutation_rate = min(0.9, 0.8 * mutation_intensity)
        add_gene_rate = min(0.3, 0.08 * mutation_intensity)
        add_node_rate = min(0.1, 0.02 * mutation_intensity)

        if random.random() < weight_mutation_rate:
            for i in range(len(self.genes)):
                self.genes[i].mutate(mutation_intensity)

        if random.random() < add_gene_rate:
            self.add_gene()
        
        if random.random() < add_node_rate:
            self.add_node()

        if mutation_intensity > 2.0:
            if random.random() < 0.15:
                self.add_gene()
            if random.random() < 0.05:
                self.add_node()

            if random.random() < 0.1:
                for gene in self.genes:
                    if random.random() < 0.1:
                        gene.enabled = not gene.enabled

    def get_node(self, n):
        for i in range(len(self.nodes)):
            if self.nodes[i].number == n:
                return self.nodes[i]
        print("Node not found : Something's Wrong ", n)
        return None

    def connect_genes(self):
        for i in range(len(self.genes)):
            self.genes[i].in_node = self.get_node(self.genes[i].in_node.number)
            self.genes[i].out_node = self.get_node(self.genes[i].out_node.number)

        for i in range(len(self.nodes)):
            self.nodes[i].in_genes.clear()

        for i in range(len(self.genes)):
            self.genes[i].out_node.in_genes.append(self.genes[i])
        pass

    def get_outputs(self, inputs):
        if len(inputs) != self.n_inputs:
            print("Wrong number of inputs")
            return [-1]

        for i in range(self.n_inputs):
            self.nodes[i].output = inputs[i]

        self.connect_genes()

        for layer in range(2, self.gh.highest_hidden + 1):
            nodes_in_layer = []
            for n in range(len(self.nodes)):
                if self.nodes[n].layer == layer:
                    nodes_in_layer.append(self.nodes[n])

            for n in range(len(nodes_in_layer)):
                nodes_in_layer[n].calculate()

        final_outputs = []
        for n in range(self.n_inputs, self.n_inputs + self.n_outputs):
            self.nodes[n].calculate()
            final_outputs.append(self.nodes[n].output)

        return final_outputs

    def get_weight(self, inno):
        for g in self.genes:
            if g.inno == inno:
                return g.weight
        return -1

    def crossover(self, partner):
        child = Genome(self.gh)
        child.nodes.clear()

        try:
            p1_highest_inno = max([(a.inno) for a in self.genes]) if self.genes else -1
        except Exception:
            p1_highest_inno = -1

        try:
            p2_highest_inno = max([(a.inno) for a in partner.genes]) if partner.genes else -1
        except Exception:
            p2_highest_inno = -1

        if self.fitness >= partner.fitness:
            child.total_nodes = self.total_nodes
            for i in range(self.total_nodes):
                child.nodes.append(self.nodes[i].clone())
        else:
            child.total_nodes = partner.total_nodes
            for i in range(partner.total_nodes):
                child.nodes.append(partner.nodes[i].clone())

        highest_inno = max(p1_highest_inno, p2_highest_inno)

        for i in range(highest_inno + 1):
            e1 = self.exists(i)
            e2 = partner.exists(i)
            
            if e1 and e2:
                if random.random() < 0.5:
                    child.genes.append(self.get_gene(i))
                else:
                    child.genes.append(partner.get_gene(i))
            elif e1 and self.fitness >= partner.fitness:
                child.genes.append(self.get_gene(i))
            elif e2 and partner.fitness > self.fitness:
                child.genes.append(partner.get_gene(i))

        child.connect_genes()
        return child

    def get_gene(self, inno):
        for g in self.genes:
            if g.inno == inno:
                return g.clone()
        print("Gene not found")
        return None

    def calculate_compatibility(self, partner, summary=False):
        try:
            p1_highest_inno = max([(a.inno) for a in self.genes])
        except Exception:
            p1_highest_inno = 0

        try:
            p2_highest_inno = max([(a.inno) for a in partner.genes])
        except Exception:
            p2_highest_inno = 0

        if self.fitness > partner.fitness:
            highest_inno = p1_highest_inno
        else:
            highest_inno = p2_highest_inno

        matching = 0
        disjoint = 0
        excess = 0

        total_weights = 0

        for i in range(highest_inno + 1):
            e1 = self.exists(i)
            e2 = partner.exists(i)
            if e1 or e2:
                if e1 and e2:
                    matching += 1
                    total_weights += abs(self.get_weight(i)) + abs(
                        partner.get_weight(i)
                    )
                    continue
                disjoint += 1

        avg_weights = total_weights / (1 if matching == 0 else matching)
        for i in range(highest_inno + 1, max(p1_highest_inno, p2_highest_inno) + 1):
            e1 = self.exists(i)
            e2 = partner.exists(i)
            if e1 or e2:
                excess += 1
            pass

        N = 1 if highest_inno < 20 else highest_inno
        excess_coeff = self.c1 * excess / N
        disjoint_coeff = self.c2 * disjoint / N
        weight_coeff = self.c3 * avg_weights

        cd = excess_coeff + disjoint_coeff + weight_coeff

        if summary:
            print("Crossover Summary:")
            print("Highest Inno:", highest_inno)
            print("Matching:", matching)
            print("Avg Weights:", avg_weights)
            print("Disjoint", disjoint)
            print("Excess:", excess)
            print("Compatibility Distance", cd)
            print("---------------------------------------")

        return cd

    def add_node(self):
        if len(self.genes) == 0:
            self.add_gene()

        if random.random() < 0.2:
            self.gh.highest_hidden += 1

        n = Node(self.total_nodes, random.randint(2, self.gh.highest_hidden))
        self.total_nodes += 1

        g = random.choice(self.genes)
        l1 = g.in_node.layer
        l2 = g.out_node.layer
        if l2 == 1:
            l2 = 1000000

        while l1 > n.layer or l2 < n.layer:
            g = random.choice(self.genes)
            l1 = g.in_node.layer
            l2 = g.out_node.layer
            if l2 == 1:
                l2 = 1000000

        self.connect_nodes(g.in_node, n)
        self.connect_nodes(n, g.out_node)
        self.genes[-1].weight = 1.0
        self.genes[-2].weight = g.weight
        g.enabled = False
        self.nodes.append(n)
        pass

    # Get Some info
    def get_info(self) -> str:
        s = f"\nGenome (Fitness: {round(self.fitness, 4)}) --------------\n"
        for g in self.genes:
            s += g.get_info()

        s += "---------------------------------------"
        return s

    def __str__(self):
        return self.get_info()

    def show(self, ds):
        ds.fill((255, 255, 255, 0))
        w, h = ds.get_size()
        vert_gap = h / (self.n_inputs + 1)
        for i in range(self.n_inputs):
            self.nodes[i].pos = [30, self.nodes[i].number * vert_gap + vert_gap]
        vert_gap = h / (self.n_outputs + 1)
        for i in range(self.n_inputs, self.n_inputs + self.n_outputs):
            self.nodes[i].pos = [
                w - 30,
                (self.nodes[i].number - self.n_inputs) * vert_gap + vert_gap,
            ]
        vert_gap = h / ((len(self.nodes) - (self.n_inputs + self.n_outputs)) + 1)
        for i in range(self.n_inputs + self.n_outputs, len(self.nodes)):
            self.nodes[i].pos = [
                w / 2,
                (self.nodes[i].number - self.n_inputs - self.n_outputs) * vert_gap
                + vert_gap,
            ]

        for g in self.genes:
            g.show(ds)
        for n in self.nodes:
            n.show(ds)
        pass