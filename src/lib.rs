use std::{cmp::Ordering, f32::consts::E};

use rand::{thread_rng, Rng};

pub const TANH: &dyn Fn(f32) -> f32 = &|x| (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x));

#[derive(Clone)]
pub struct Agent<'a> {
    inputs: i32,
    nodes: i32,
    connections: i32,
    outputs: i32,
    data_lists: Vec<Vec<f32>>,
    connection_list: Vec<Connection>,
    activation_funcs: Vec<&'a dyn Fn(f32) -> f32>,
}

#[derive(Clone, Copy)]
pub struct Connection {
    start_layer: usize,
    end_layer: usize,
    start_idx: usize,
    end_idx: usize,
    weight: f32,
}

impl Agent<'_> {
    pub fn create_agents(
        amount: i32,
        inputs: i32,
        outputs: i32,
        activation_funcs: Vec<&'static dyn Fn(f32) -> f32>,
    ) -> Vec<Self> {
        let mut result: Vec<Self> = vec![];

        for _ in 0..amount {
            result.push(Agent {
                inputs,
                nodes: 0,
                connections: 0,
                outputs,
                data_lists: vec![
                    vec![0.0; inputs.try_into().unwrap()],
                    vec![],
                    vec![0.0; outputs.try_into().unwrap()],
                ],
                connection_list: vec![],
                activation_funcs: activation_funcs.clone(),
            })
        }

        return result;
    }

    pub fn calculate(&mut self, input: &Vec<f32>) -> Vec<f32> {
        if input.len() != self.inputs.try_into().unwrap() {
            panic!(
                "Input size ({}) doesn't match target input size ({})",
                input.len(),
                self.inputs
            );
        } else {
            self.data_lists[0] = input.to_vec();
        }

        for idx in 0..self.data_lists[2].len() {
            self.data_lists[2][idx] = 0.0;
        }

        self.data_lists[1].clear();

        for _ in 0..self.nodes {
            self.data_lists[1].push(0.0);
        }

        self.sort_connections();

        for connection in &self.connection_list {
            self.data_lists[connection.end_layer][connection.end_idx] += (self.activation_funcs
                [connection.start_layer])(
                self.data_lists[connection.start_layer][connection.start_idx],
            ) * connection.weight;
        }

        return self.data_lists[2].clone();
    }

    pub fn sort_connections(&mut self) {
        self.connection_list
            .sort_by(|a, b| match a.start_layer.cmp(&b.start_layer) {
                Ordering::Equal => a.end_layer.cmp(&b.end_layer),
                other => other,
            });
    }

    pub fn reproduce(
        &self,
        new_node_chance: f32,
        new_connection_chance: f32,
        delete_node_chance: f32,
        delete_connection_chance: f32,
        change_weight_chance: f32,
        change_connection_chance: f32,
        max_weight: f32,
    ) -> Self {
        let mut new_agent = Agent {
            inputs: self.inputs,
            nodes: self.nodes,
            connections: self.connections,
            outputs: self.outputs,
            data_lists: self.data_lists.clone(),
            connection_list: self.connection_list.clone(),
            activation_funcs: self.activation_funcs.clone(),
        };
        let mut rng = thread_rng();

        if rng.gen_range(0.0..1.0) < delete_node_chance && new_agent.nodes > 0 {
            let idx = rng.gen_range(0..new_agent.nodes);
            new_agent.nodes -= 1;

            for connection in new_agent.connection_list.iter_mut() {
                if connection.start_layer == 1 && connection.start_idx >= idx.try_into().unwrap() {
                    if connection.start_idx == idx.try_into().unwrap() {
                        if new_agent.nodes > 0 {
                            connection.start_idx =
                                rng.gen_range(0..new_agent.nodes).try_into().unwrap();
                        } else {
                            connection.start_layer = 0;
                            connection.start_idx =
                                rng.gen_range(0..new_agent.inputs).try_into().unwrap();
                        }
                    } else {
                        connection.start_idx -= 1;
                    }
                }

                if connection.end_layer == 1 && connection.end_idx >= idx.try_into().unwrap() {
                    if connection.end_idx == idx.try_into().unwrap() {
                        if new_agent.nodes > 0 {
                            connection.end_idx =
                                rng.gen_range(0..new_agent.nodes).try_into().unwrap();
                        } else {
                            connection.end_layer = 2;
                            connection.end_idx =
                                rng.gen_range(0..new_agent.outputs).try_into().unwrap();
                        }
                    } else {
                        connection.end_idx -= 1;
                    }
                }
            }

            new_agent.data_lists[1].pop();
        }

        if rng.gen_range(0.0..1.0) < new_node_chance {
            new_agent.nodes += 1;

            new_agent.data_lists[1].push(0.0);
        }

        if rng.gen_range(0.0..1.0) < delete_connection_chance && new_agent.connections > 0 {
            let idx = rng.gen_range(0..new_agent.connections);

            new_agent.connections -= 1;

            new_agent.connection_list.remove(idx.try_into().unwrap());
        }

        if rng.gen_range(0.0..1.0) < new_connection_chance {
            new_agent.connections += 1;

            if new_agent.nodes > 0 {
                let start_layer = rng.gen_range(0..=1);
                let start_idx;

                if start_layer == 0 {
                    start_idx = rng.gen_range(0..new_agent.inputs);
                } else {
                    start_idx = rng.gen_range(0..new_agent.nodes);
                }

                let end_layer = rng.gen_range(1..=2);
                let end_idx;

                if end_layer == 1 {
                    end_idx = rng.gen_range(0..new_agent.nodes);
                } else {
                    end_idx = rng.gen_range(0..new_agent.outputs);
                }

                let new_connection = Connection {
                    start_layer,
                    end_layer,
                    start_idx: start_idx.try_into().unwrap(),
                    end_idx: end_idx.try_into().unwrap(),
                    weight: rng.gen_range(-max_weight..max_weight),
                };

                new_agent.connection_list.push(new_connection);
            } else {
                let start_layer = 0;
                let start_idx;

                start_idx = rng.gen_range(0..new_agent.inputs);

                let end_layer = 2;
                let end_idx;

                end_idx = rng.gen_range(0..new_agent.outputs);

                let new_connection = Connection {
                    start_layer,
                    end_layer,
                    start_idx: start_idx.try_into().unwrap(),
                    end_idx: end_idx.try_into().unwrap(),
                    weight: rng.gen_range(-max_weight..max_weight),
                };

                new_agent.connection_list.push(new_connection);
            }
        }

        if rng.gen_range(0.0..1.0) < change_connection_chance && new_agent.connections > 0 {
            let idx: usize = rng.gen_range(0..new_agent.connections).try_into().unwrap();

            if new_agent.nodes > 0 {
                let new_start_layer: usize = rng.gen_range(0..=1);
                let new_end_layer: usize = rng.gen_range(1..=2);

                new_agent.connection_list[idx].start_layer = new_start_layer;
                new_agent.connection_list[idx].end_layer = new_end_layer;
            } else {
                let new_start_layer: usize = 0;
                let new_end_layer: usize = 2;

                new_agent.connection_list[idx].start_layer = new_start_layer;
                new_agent.connection_list[idx].end_layer = new_end_layer;
            }

            if new_agent.connection_list[idx].start_layer == 0 {
                let start_idx: usize = rng.gen_range(0..new_agent.inputs).try_into().unwrap();

                new_agent.connection_list[idx].start_idx = start_idx;
            } else {
                let start_idx: usize = rng.gen_range(0..new_agent.nodes).try_into().unwrap();

                new_agent.connection_list[idx].start_idx = start_idx;
            }

            if new_agent.connection_list[idx].end_layer == 1 {
                let end_idx: usize = rng.gen_range(0..new_agent.nodes).try_into().unwrap();

                new_agent.connection_list[idx].end_idx = end_idx;
            } else {
                let end_idx: usize = rng.gen_range(0..new_agent.outputs).try_into().unwrap();

                new_agent.connection_list[idx].end_idx = end_idx;
            }
        }

        if rng.gen_range(0.0..1.0) < change_weight_chance && new_agent.connections > 0 {
            let idx: usize = rng.gen_range(0..new_agent.connections).try_into().unwrap();

            new_agent.connection_list[idx].weight = rng.gen_range(-max_weight..max_weight);
        }

        return new_agent;
    }

    pub fn print(&mut self) {
        self.sort_connections();
        println!("Nodes: {} ", self.nodes);
        println!("Connections: {} ", self.connections);
        println!();

        for idx in 0..self.connection_list.len() {
            println!("Connection: {}", idx);
            println!(
                "  From   : {}, {}",
                self.connection_list[idx].start_layer, self.connection_list[idx].start_idx
            );
            println!(
                "  To     : {}, {}",
                self.connection_list[idx].end_layer, self.connection_list[idx].end_idx
            );
            println!("  Weight : {}", self.connection_list[idx].weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_use() {
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

            let index_of_max: usize = result
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
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
}
