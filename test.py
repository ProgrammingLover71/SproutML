from src import *


def main() -> None:

    gen = Genome(2, 1, 'tanh')
    
    # Add 2 nodes
    gen.add_node(1, 3)
    gen.add_node(2, 3)

    print(gen.nodes)
    print(gen.conns)


if __name__ == "__main__":
    main()