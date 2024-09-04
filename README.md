# simple_neat

**simple_neat** is a lightweight Rust library designed to implement the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. This library provides essential tools for evolving neural networks, supporting dynamic topologies and enabling the evolution of both the structure and weights of the network.

## Features

- **Dynamic Network Topology**: Supports the evolution of neural networks by adding or removing nodes and connections.
- **Customizable Activation Functions**: Allows you to define and use custom activation functions.
- **Reproduction and Mutation**: Implements key NEAT operations such as reproduction, mutation (adding/removing nodes and connections), and weight adjustments.
- **Sorting and Calculating Network Outputs**: Provides functionalities to sort connections and compute network outputs based on the current topology.

## Getting Started

### Installation

Add `simple_neat` to your `Cargo.toml`:

```toml
[dependencies]
simple_neat = "0.1.0"
```

### Example Usage

Here is a simple example demonstrating how to create agents, run them through multiple epochs, and evolve the top-performing agents:

```rust
use simple_neat::{Agent, TANH};
use rand::thread_rng;

fn main() {
    let mut agents = Agent::create_agents(5, 2, 1, vec![TANH, TANH]);

    let mut rng = thread_rng();
    const EPOCHS: i32 = 10_000;

    for epoch in 0..EPOCHS {
        println!("Epoch: {}", epoch);

        let input = vec![rng.gen_range(-1.0..1.0), 1.0];
        let mut result: Vec<f32> = vec![];

        for agent in agents.iter_mut() {
            result.push(agent.calculate(&input)[0] / (agent.nodes + 1) as f32);
        }

        let index_of_max = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap();

        let best_agent = agents.remove(index_of_max);
        agents = vec![best_agent];
        println!("Best result: {}", result[index_of_max]);

        for _ in 0..4 {
            agents.push(agents[0].reproduce(0.1, 0.15, 0.05, 0.05, 0.20, 0.15, 3.0));
        }
    }

    agents[0].print();
}
```

### Key Functions

- **`Agent::create_agents`**: Creates a vector of agents with the specified number of inputs, outputs, and activation functions.
- **`Agent::calculate`**: Computes the output of the network based on the given inputs.
- **`Agent::reproduce`**: Generates a new agent by applying mutations to the parent agent.
- **`Agent::print`**: Outputs the structure of the network, including nodes and connections.

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to report bugs or suggest features.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
