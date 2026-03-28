##============ Imports ===========##


from typing import Callable, Literal

import copy
import numpy as np
import random
from .activations import *


##============ Node & Connection Genes ============##


node_type = Literal['input', 'hidden', 'output']


class NodeGene:
    """
    The NodeGene class holds all necessary information for a node in a NEAT genome.
    """

    def __init__(self, id: int, type: node_type, activation: activation_func = 'sigmoid') -> None:
        """
        Initializes the node gene with the specified ID, node type and activation function.

        Args:
            id: The ID of this node.
            type: The type of the node ('input', 'hidden' or 'output').
            activation: The activation function for this node ('sigmoid', 'relu', or 'tanh').
        """

        self.id: int         = id
        self.type: node_type = type
        self.activation      = get_activation_function(activation)
    

    def __repr__(self) -> str:
        return f"[{self.type}(id={self.id}) - A: {self.activation.__name__}]"


class ConnectionGene:
    """
    The ConnectionGene class holds all information for a connection between two nodes in a NEAT genome.
    """
    
    def __init__(self, in_id: int, out_id: int, weight: float, innovation: int, enabled: bool = True) -> None:
        """
        Initializes the connection gene with the specified IDs, weight and innovation number.

        Args:
            in_id: The ID of the input node of the connection.
            out_id: The ID of the output node of the connection.
            weight: The weight of the node, determining how much the signal is transmitted forward.
            innovation: The innovation number of the connection. Relevant in genome crossover.
            enabled: Whether the weight is initially enabled.
        """

        self.in_id      = in_id
        self.out_id     = out_id
        self.weight     = weight
        self.innovation = innovation
        self.enabled    = enabled
    
    
    def __repr__(self) -> str:
        return f"< innovation:{self.innovation}, {self.in_id}->{self.out_id}, w:{self.weight} {'enabled' if self.enabled else 'disabled'} >"


##============ Innovation Number and Node ID counter ============##


_global_node_id_counter: int = 1000

_global_innovation_counter: int = 0
_global_innovation_map: dict[tuple[int, int], int] = {}


def new_innovation_number(in_id: int, out_id: int) -> int:
    """
    Returns a new, unique innovation number based on the input and output node IDs.

    Args:
        in_id: The ID of the input node for the connection.
        out_id: The ID of the output node for the connection.

    Returns:
        An `int` representing a new innovation number for a connection based on the input and output node IDs.
    """

    global _global_innovation_counter, _global_innovation_map

    if (in_id, out_id) not in _global_innovation_map:

        _global_innovation_counter += 1
        _global_innovation_map[(in_id, out_id)] = _global_innovation_counter
    
    return _global_innovation_map[(in_id, out_id)]


def new_node_id() -> int:
    """
    Returns a new, unique node ID.

    Returns:
        An `int` representing unique node ID.
    """

    global _global_node_id_counter

    _global_node_id_counter += 1
    return _global_node_id_counter


def reset_innovation_counters() -> None:
    """Resets innovation/node counters for deterministic reproducibility."""
    global _global_node_id_counter, _global_innovation_counter, _global_innovation_map
    
    _global_node_id_counter = 1000
    _global_innovation_counter = 0
    _global_innovation_map = {}


##============ Genomes ============##


class Genome:
    """
    The Genome class represents a NEAT genome, storing information about its nodes, connections and fitness.
    """

    def __init__(self, num_inputs: int, num_outputs: int, activation: activation_func):
        """
        Initializes the genome with default values for the given structure.

        Args:
            num_inputs: The number of inputs for the genome.
            num_outputs: The number of outputs for the genome.
        """

        self._num_inputs: int             = num_inputs
        self._num_outputs: int            = num_outputs
        self._activation: activation_func = activation

        self.nodes: dict[int, NodeGene]         = {}
        self.conns: dict[int, ConnectionGene]   = {}

        # Add nodes with fixed IDs
        for i in range(num_inputs):
            self.nodes[i] = NodeGene(i, 'input', activation)
        
        for i in range(num_outputs):
            self.nodes[num_inputs + i] = NodeGene(num_inputs + i, 'output', activation)

        # Separate node IDs
        input_ids = [nid for nid, n in self.nodes.items() if n.type == "input"]
        output_ids = [nid for nid, n in self.nodes.items() if n.type == "output"]

        # Fully connect inputs → outputs
        for in_id in input_ids:
            for out_id in output_ids:
                innovation = new_innovation_number(in_id, out_id)

                self.conns[innovation] = ConnectionGene(
                    in_id = in_id,
                    out_id = out_id,
                    weight = np.random.uniform(-1, 1),  # random initial weights
                    innovation = innovation,
                    enabled = True
                )

        
        self.fitness: float = float("-inf")
    
    #------------------------------------------------------------------------------------------------#

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Feeds the specified inputs through the genome, returning the its output.

        Args:
            inputs: The inputs to feed the genome.

        Returns:
            A `list[float]` containing the outputs of the genome.
        """

        values: dict[int, float] = {node_id: 0.0 for node_id in self.nodes}

        # 1. Stable input ordering
        input_nodes = sorted(
            (n for n in self.nodes.values() if n.type == "input"),
            key=lambda n: n.id
        )

        if len(inputs) != len(input_nodes):
            raise ValueError("Input size mismatch")

        for node, val in zip(input_nodes, inputs):
            values[node.id] = val

        # 2. Build incoming connection map
        incoming: dict[int, list[ConnectionGene]] = {nid: [] for nid in self.nodes}

        for conn in self.conns.values():

            if conn.enabled:
                incoming[conn.out_id].append(conn)

        # 3. Topological order (cycles are handled implicitly)
        order = self.topological_sort(allow_cycles=True)

        def update_values() -> float:
            max_delta = 0.0
            for node_id in order:
                node = self.nodes[node_id]
                if node.type == "input":
                    continue

                total = 0.0
                for conn in incoming[node_id]:
                    total += values[conn.in_id] * conn.weight

                new_val = float(node.activation(total))
                max_delta = max(max_delta, abs(new_val - values[node_id]))
                values[node_id] = new_val

            return max_delta

        # 4. Forward pass with recurrent convergence on cycles
        update_values()  # first pass
        for _ in range(9):  # additional iterations for cyclic paths
            if update_values() < 1e-6:
                break

        # 5. Stable output ordering
        output_nodes = sorted(
            (n for n in self.nodes.values() if n.type == "output"),
            key=lambda n: n.id
        )

        return [values[n.id] for n in output_nodes]
    
    #------------------------------------------------------------------------------------------------#

    def topological_sort(self, allow_cycles: bool = True) -> list[int]:
        """
        Topologically sorts the nodes of the network.
        If a cycle exists and allow_cycles=True, the remaining nodes are appended in numerical order.
        """

        in_degree = {nid: 0 for nid in self.nodes}

        for conn in self.conns.values():
            if conn.enabled:
                in_degree[conn.out_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[int] = []

        while queue:

            node = queue.pop(0)
            order.append(node)

            for conn in self.conns.values():

                if conn.enabled and conn.in_id == node:
                    in_degree[conn.out_id] -= 1

                    if in_degree[conn.out_id] == 0:
                        queue.append(conn.out_id)

        if len(order) != len(self.nodes):
            if not allow_cycles:
                raise RuntimeError("Cycle detected in network")

            remaining = sorted(set(self.nodes.keys()) - set(order))
            order.extend(remaining)

        return order
    
    #------------------------------------------------------------------------------------------------#

    def add_node(self, in_node: int, out_node: int) -> None:
        """
        Splits an enabled connection between two nodes, inserting a hidden node and two new connections.

        Args:
            in_node (int): The ID of the start node.
            out_node (int): The ID of the end node.
        """

        if in_node == out_node:
            raise ValueError("Cannot add node on self-loop")

        old_innovation = new_innovation_number(in_node, out_node)

        if old_innovation not in self.conns or not self.conns[old_innovation].enabled:
            raise ValueError(f"Connection {in_node}->{out_node} is missing or disabled and cannot be split")

        old_connection = self.conns[old_innovation]

        new_node = NodeGene(new_node_id(), 'hidden', self._activation)

        # split the connection
        inn1 = new_innovation_number(in_node, new_node.id)
        conn1 = ConnectionGene(in_node, new_node.id, old_connection.weight, inn1)

        inn2 = new_innovation_number(new_node.id, out_node)
        conn2 = ConnectionGene(new_node.id, out_node, 1.0, inn2)

        self.nodes[new_node.id] = new_node
        self.conns[inn1] = conn1
        self.conns[inn2] = conn2

        old_connection.enabled = False
    

    def add_connection(self, in_node: int, out_node: int) -> None:
        """
        Adds a new (or re-enables existing) connection between two nodes.
        """
        if in_node == out_node:
            return

        if in_node not in self.nodes or out_node not in self.nodes:
            raise ValueError("Node IDs must exist in the genome")

        if self.nodes[out_node].type == 'input':
            # Do not connect into input nodes
            return

        innovation = new_innovation_number(in_node, out_node)

        if innovation in self.conns:
            self.conns[innovation].enabled = True
            return

        self.conns[innovation] = ConnectionGene(in_node, out_node, np.random.uniform(-1, 1), innovation)
    
    #------------------------------------------------------------------------------------------------#

    @staticmethod
    def crossover(gen1: Genome, gen2: Genome) -> Genome:
        """
        Reproduce (or cross over) a genome with another one.

        Args:
            gen1 (Genome): The first genome to participate in the crossover.
            gen2 (Genome): The second genome to participate in the crossover.

        Returns:
            A `Genome` representing the child of the two genomes.
        """

        fittest = gen1 if gen1.fitness >= gen2.fitness else gen2

        # gen1 and gen2 have the same input/output sizes, so we take gen1's (this is completely arbitrary)
        child = Genome(gen1._num_inputs, gen1._num_outputs, fittest._activation)

        all_innovations = set(gen1.conns.keys()).union(gen2.conns.keys())
        g1_genes = gen1.conns
        g2_genes = gen2.conns

        # STEP 1: Inherit connections
        # ------------ #
        for inn in sorted(all_innovations):
            gene1 = g1_genes.get(inn)
            gene2 = g2_genes.get(inn)

            # Case 1 -- Matching genes
            if gene1 and gene2:

                chosen = random.choice([gene1, gene2])

                if not gene1.enabled or not gene2.enabled:
                    enabled = np.random.uniform() > (3 / 4)     # 75% chance to stay off
                
                else:
                    enabled = True
                
                # Create the new gene
                child.conns[inn] = ConnectionGene(
                    in_id = chosen.in_id,
                    out_id = chosen.out_id,
                    weight = chosen.weight,
                    innovation = inn,
                    enabled = enabled
                )

            # Cases 2 & 3 -- Disjoint or Excess genes
            elif gene1 and not gene2:
                if fittest == gen1:
                    child.conns[inn] = ConnectionGene(
                        in_id = gene1.in_id,
                        out_id = gene1.out_id,
                        weight = gene1.weight,
                        innovation = gene1.innovation,
                        enabled = gene1.enabled
                    )

            elif gene2 and not gene1:
                if fittest == gen2:
                    child.conns[inn] = ConnectionGene(
                        in_id = gene2.in_id,
                        out_id = gene2.out_id,
                        weight = gene2.weight,
                        innovation = gene2.innovation,
                        enabled = gene2.enabled
                    )

        # STEP 2: Inherit nodes
        # ------------ #
        node_ids: set[int] = set()
        for conn in child.conns.values():
            node_ids.add(conn.in_id)
            node_ids.add(conn.out_id)

        # Add hidden nodes from parents
        for gen in [gen1, gen2]:

            for nid, node in gen.nodes.items():
                if node.type == 'hidden':
                    node_ids.add(nid)

        # Build child nodes
        for nid in node_ids:
            if nid in child.nodes:
                continue  # input/output already set

            n1 = gen1.nodes.get(nid)
            n2 = gen2.nodes.get(nid)

            if n1 and n2:
                # Matching node: pick from fitter (or random if equal)
                if gen1.fitness == gen2.fitness:
                    chosen_node = random.choice([n1, n2])
                else:
                    chosen_node = n1 if fittest == gen1 else n2

            elif n1:
                chosen_node = n1 if fittest == gen1 else n1  # fallback anyway

            elif n2:
                chosen_node = n2 if fittest == gen2 else n2

            else:
                continue  # shouldn't happen

            if chosen_node.type == 'hidden':  # only add hidden nodes from parents
                child.nodes[nid] = NodeGene(
                    id = nid,
                    type = chosen_node.type,
                    activation = child._activation
                )

        return child

    def mutate(self, mutation_rate: float = 0.05, add_connection_rate: float = 0.05, add_node_rate: float = 0.03, weight_perturb_scale: float = 0.5) -> None:
        """Mutates this genome in place."""

        # Weight perturbation
        for conn in self.conns.values():
            if random.random() < mutation_rate:
                conn.weight += np.random.normal(0, weight_perturb_scale)

        # Add a new connection
        if random.random() < add_connection_rate and len(self.nodes) >= 2:
            possible_in = [nid for nid, n in self.nodes.items() if n.type != 'output']
            possible_out = [nid for nid, n in self.nodes.items() if n.type != 'input']

            if possible_in and possible_out:
                in_node = random.choice(possible_in)
                out_node = random.choice(possible_out)

                if in_node != out_node and not any(c.in_id == in_node and c.out_id == out_node for c in self.conns.values()):
                    self.add_connection(in_node, out_node)

        # Add a new node by splitting an existing connection
        if random.random() < add_node_rate:
            enabled_conns = [c for c in self.conns.values() if c.enabled]
            if enabled_conns:
                conn = random.choice(enabled_conns)
                # Ensure we don't add a node on a disabled connection
                self.add_node(conn.in_id, conn.out_id)


##============ Species ===========##


class Species:
    """
    The Species class represents a genome species in NEAT.

    A <i>species</i> is a group of genomes that have similar genes in their structure.
    """

    def __init__(self, genomes: list[Genome] = []) -> None:
        self.genomes = genomes
    
    
    def add_genome(self, genome: Genome) -> None:
        """Adds a genome to the species."""
        self.genomes.append(genome)

    
    def fitness(self) -> float:
        """Returns the fitness of this species."""
        if not self.genomes:
            return 0.0

        # Species adjust fitness: larger ones are penalized, while smaller ones are boosted
        return sum(gen.fitness for gen in self.genomes) / len(self.genomes)
    

    def representative(self) -> Genome:
        """
        Returns the representative of this species.
        The representative is the genome with the highest fitness.
        """
        if not self.genomes:
            raise ValueError("Species contains no genomes")

        best = self.genomes[0]
        max_fit = best.fitness

        for gen in self.genomes[1:]:
            if gen.fitness > max_fit:
                max_fit = gen.fitness
                best = gen

        return best


    @staticmethod
    def compatibility(gen1: Genome, gen2: Genome, c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
        """
        Returns a float indicating the compatibility between two genomes.

        Args:
            gen1 (Genome): The first genome.
            gen2 (Genome): The second genome.
            c1 (float, optional - default 1.0): The constant multiplier for excess genes.
            c2 (float, optional - default 1.0): The constant multiplier for disjoint genes.
            c3 (float, optional - default 0.4): The constant multiplier for the average weight difference.
        
        Returns:
            A `float` between 0 (incompatible) and 1 (identical) representing the compatibility between the two genomes.
        """
        
        g1_genes = gen1.conns
        g2_genes = gen2.conns

        all_gene_inns = set(g1_genes.keys()).union(set(g2_genes.keys()))

        matching: list[float] = []
        disjoint: int = 0
        excess: int   = 0

        max1 = max(g1_genes.keys(), default = 0)
        max2 = max(g2_genes.keys(), default = 0)

        for gene in all_gene_inns:

            g1_gene = g1_genes.get(gene)
            g2_gene = g2_genes.get(gene)

            # If both genes exist with the same innovation number (I.N.)
            if g1_gene and g2_gene:
                matching.append(
                    abs(g1_gene.weight - g2_gene.weight)
                )
            
            # Excess and disjoint genes
            elif gene > max(max1, max2):
                excess += 1
            
            else:
                disjoint += 1
        
        # N = number of genes
        N = max(len(g1_genes.keys()), len(g2_genes.keys()))
        if N < 1: N = 1     # type: ignore
        
        # W = mean weight difference
        W = float(np.mean(matching)) if matching else 0

        # Final formula:
        # d' = [(c1 * excess / N) + (c2 * disjoint / N) + (c3 * W)]
        # d = 1 - d
        dp = (c1 * excess / N) + (c2 * disjoint / N) + c3 * W
        return 1 - dp


    @staticmethod
    def speciate(population: list[Genome], species_threshold: float) -> list[Species]:
        """
        Speciates a population of genomes based on a species threshold.

        Args:
            population (list[Genome]): The population of genomes.
            species_threshold (float): The minimum compatibility threshold for two genomes to be in the same species. Must be between 0.0 and 1.0.

        Returns:
            A `list[Species]` containing all identified species.
        """

        species: list[Species] = []

        for genome in population:

            placed = False
            for spec in species:

                representative = spec.representative()
                dist = Species.compatibility(genome, representative)

                if dist > species_threshold:

                    spec.add_genome(genome)
                    placed = True
                    break
                
            if not placed:
                species.append(Species([genome]))
        
        return species
    

##============ Ppoulations ============##


class Population:
    """
    Represents a population of genomes in NEAT.
    """

    def __init__(self, num_inputs: int, num_outputs: int, activation: activation_func, num_genomes: int = 100):

        # Create the initial conditions
        self.genomes: list[Genome] = []
        
        for _ in range(num_genomes):
            self.genomes.append(Genome(num_inputs, num_outputs, activation))
    

    def fittest(self) -> Genome:
        """
        Returns the fittest genome in the entire population.
        """
        if not self.genomes:
            raise ValueError("Population contains no genomes")

        best = self.genomes[0]
        max_fit = best.fitness

        for gen in self.genomes[1:]:
            if gen.fitness > max_fit:
                max_fit = gen.fitness
                best = gen

        return best
        

##============ Evolution Loop ============##


def evolve(pop: Population, fitness: Callable[[Genome], float], num_epochs: int = 100, mutation_rate: float = 0.05) -> Population:
    """
    Evolves a population of genomes over a specified amount of epochs using a simple NEAT cycle.

    Args:
        pop (Population): The population to evolve.
        num_epochs (int, optional - default 100): How many epochs to evolve the population for.
        mutation_rate (float, optional - default 0.05): Base mutation chance when altering weights.

    Returns:
        The final `Population` after `num_epochs` epochs of evolution.
    """

    num_genomes = len(pop.genomes)

    for _ in range(num_epochs):

        # Evaluate fitness for all genomes
        for gen in pop.genomes:
            gen.fitness = fitness(gen)

        # Sort by fitness descending
        sorted_genomes = sorted(pop.genomes, key=lambda g: g.fitness, reverse=True)

        # Elitism: keep top 2
        new_genomes: list[Genome] = []
        if num_genomes > 0:
            new_genomes.append(copy.deepcopy(sorted_genomes[0]))
        if num_genomes > 1:
            new_genomes.append(copy.deepcopy(sorted_genomes[1]))

        # Reproduction loop
        while len(new_genomes) < num_genomes:

            # Tournament selection
            parents = random.sample(sorted_genomes[:max(2, len(sorted_genomes) // 2)], k=2) if len(sorted_genomes) >= 2 else [sorted_genomes[0], sorted_genomes[0]]
            child = Genome.crossover(parents[0], parents[1])

            # Mutate child
            child.mutate(mutation_rate=mutation_rate)

            new_genomes.append(child)

        pop.genomes = new_genomes

    # Final evaluation to keep fitness up-to-date
    for gen in pop.genomes:
        gen.fitness = fitness(gen)

    return pop
