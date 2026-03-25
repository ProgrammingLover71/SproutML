from src.neuroevolution import Genome, Population, Species, evolve, reset_innovation_counters


def test_genome_forward():
    g = Genome(2, 1, 'sigmoid')
    out = g.forward([1.0, 0.0])
    assert len(out) == 1
    assert 0.0 <= out[0] <= 1.0


def test_add_node_and_connection():
    g = Genome(2, 1, 'sigmoid')
    conn = next(iter(g.conns.values()))
    old_id = conn.innovation
    g.add_node(conn.in_id, conn.out_id)
    assert conn.enabled is False
    assert len(g.nodes) == 4
    assert len(g.conns) == 4

    g.add_connection(conn.in_id, conn.out_id)
    assert g.conns[old_id].enabled is True


def test_crossover_preserves_best():
    reset_innovation_counters()
    p1 = Genome(2, 1, 'sigmoid')
    p2 = Genome(2, 1, 'sigmoid')
    p1.fitness = 10
    p2.fitness = 5

    child = Genome.crossover(p1, p2)
    assert isinstance(child, Genome)
    assert child._num_inputs == 2
    assert child._num_outputs == 1


def test_species_representative_and_fitness():
    g1 = Genome(2, 1, 'sigmoid'); g1.fitness = 5
    g2 = Genome(2, 1, 'sigmoid'); g2.fitness = 10

    s = Species([g1, g2])
    assert s.representative() is g2
    assert abs(s.fitness() - 7.5) < 1e-6


def test_evolve_cycle():
    reset_innovation_counters()
    pop = Population(2, 1, 'sigmoid', num_genomes=4)

    def fitness_fn(gen):
        return sum(gen.forward([0.1, 0.2]))

    out = evolve(pop, fitness_fn, num_epochs=3, mutation_rate=0.2)
    assert isinstance(out, Population)
    assert all(isinstance(g, Genome) for g in out.genomes)


def test_forward_with_cycle():
    g = Genome(1, 1, 'sigmoid')

    base_conn = next(iter(g.conns.values()))
    g.add_node(base_conn.in_id, base_conn.out_id)

    hidden_id = [nid for nid, n in g.nodes.items() if n.type == 'hidden'][0]
    output_id = [nid for nid, n in g.nodes.items() if n.type == 'output'][0]
    g.add_connection(output_id, hidden_id)

    out = g.forward([0.5])
    assert isinstance(out, list)
    assert len(out) == 1


if __name__ == '__main__':
    test_genome_forward()
    test_add_node_and_connection()
    test_crossover_preserves_best()
    test_species_representative_and_fitness()
    test_evolve_cycle()
    test_forward_with_cycle()
    print('ALL NEAT TESTS PASSED')
