from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import random

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=10, type=int)
    parser.add_argument("--T", default=100, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=32, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--max-included-literals", default=4, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

print("Creating training data")

# Create train data

graphs_train = Graphs(
    args.number_of_examples,
    symbols=['A', 'B'],
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
)

for graph_id in range(args.number_of_examples):
    graphs_train.set_number_of_graph_nodes(graph_id, 2)

graphs_train.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = 1
        graphs_train.add_graph_node(graph_id, node_id, number_of_edges)

graphs_train.prepare_edge_configuration()

Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
    edge_type = "Plain"
    source_node_id = 0
    destination_node_id = 1
    graphs_train.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    source_node_id = 1
    destination_node_id = 0
    graphs_train.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    x1 = random.choice(['A', 'B'])
    x2 = random.choice(['A', 'B'])
    if x1 == x2:
        Y_train[graph_id] = 0
    else:
        Y_train[graph_id] = 1

    graphs_train.add_graph_node_feature(graph_id, 0, x1)
    graphs_train.add_graph_node_feature(graph_id, 1, x2)

    if np.random.rand() <= args.noise:
        Y_train[graph_id] = 1 - Y_train[graph_id]

graphs_train.encode()

# Create test data

print("Creating testing data")

graphs_test = Graphs(args.number_of_examples, init_with=graphs_train)

for graph_id in range(args.number_of_examples):
    graphs_test.set_number_of_graph_nodes(graph_id, 2)

graphs_test.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        number_of_edges = 1
        graphs_test.add_graph_node(graph_id, node_id, number_of_edges)

graphs_test.prepare_edge_configuration()

Y_test = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
    edge_type = "Plain"
    source_node_id = 0
    destination_node_id = 1
    graphs_test.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    source_node_id = 1
    destination_node_id = 0
    graphs_test.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    x1 = random.choice(['A', 'B'])
    x2 = random.choice(['A', 'B'])
    if x1 == x2:
        Y_test[graph_id] = 0
    else:
        Y_test[graph_id] = 1

    graphs_test.add_graph_node_feature(graph_id, 0, x1)
    graphs_test.add_graph_node_feature(graph_id, 1, x2)

graphs_test.encode()

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth = args.depth,
    message_size = args.message_size,
    message_bits = args.message_bits,
    max_included_literals = args.max_included_literals
)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))

        # for k in range(args.message_size * 2):
        #     if tm.ta_action(1, i, k):
        #         if k < args.message_size:
        #             l.append("c%d" % (k))
        #         else:
        #             l.append("NOT c%d" % (k - args.message_size))

        print(" AND ".join(l))

print(graphs_test.hypervectors)
print(tm.hypervectors)
print(graphs_test.edge_type_id)
