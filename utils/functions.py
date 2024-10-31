import torch
import numpy as np
import random
def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)
    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    result = tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))
    return result

def get_outer_gradients(outer_loss, params, hparams, retain_graph=False):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=True)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)
    return grad_outer_w, grad_outer_hparams

def comm_delay(DELAY):
    return np.random.lognormal(np.log(DELAY), np.log(DELAY)/4) / 1000.0

def data_a_batch(batch_data):
    new_batch = []
    for i in range(len(batch_data)):
        new_batch.append(batch_data[i].squeeze(0))
    return new_batch

def sample_with_constraints(nodes, t, total_rounds, nb_select):
    unselected_counts = {node: 0 for node in nodes}
    selected_history = []
    results = []
    for round_num in range(1, total_rounds + 1):
        while True:
            selected_nodes = random.sample(nodes, nb_select)
            temp_unselected_counts = unselected_counts.copy()
            for node in nodes:
                if node in selected_nodes:
                    temp_unselected_counts[node] = 0
                else:
                    temp_unselected_counts[node] += 1
            if all(count < t for count in temp_unselected_counts.values()):
                unselected_counts = temp_unselected_counts
                selected_history.append(selected_nodes)
                results.append(selected_nodes)
                break
    return results