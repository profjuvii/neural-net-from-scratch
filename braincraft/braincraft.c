#include "braincraft.h"


void init_neuron(Neuron* neuron, int input_size) {
    neuron->weights = (double*)calloc(input_size, sizeof(double));
    for (int i = 0; i < input_size; ++i) {
        neuron->weights[i] = (double)rand() / RAND_MAX;
    }
    neuron->bias = (double)rand() / RAND_MAX;
}


void init_layer(Layer* layer, int num_neurons, int input_size) {
    layer->neurons = (Neuron*)calloc(num_neurons, sizeof(Neuron));
    for (int i = 0; i < num_neurons; ++i) {
        init_neuron(&layer->neurons[i], input_size);
    }
    layer->num_neurons = num_neurons;
}


void free_network(Layer* network, int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        for (int j = 0; j < network[i].num_neurons; ++j) {
            free(network[i].neurons[j].weights);
        }
        free(network[i].neurons);
    }
    free(network);
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double compute_neuron_output(Neuron* neuron, double* inputs, int input_size) {
    double sum = 0.0;
    for (int i = 0; i < input_size; ++i) {
        sum += neuron->weights[i] * inputs[i];
    }
    sum += neuron->bias;
    neuron->output = sigmoid(sum);
    return neuron->output;
}


double* get_activations(Neuron* neurons, int num_neurons) {
    double* activations = (double*)calloc(num_neurons, sizeof(double));
    for (int j = 0; j < num_neurons; ++j) {
        activations[j] = neurons[j].output;
    }
    return activations;
}


void forward(Layer* network, int num_layers, double* inputs, int input_size) {
    double* prev_activations = NULL;
    for (int i = 0; i < num_layers; ++i) {
        double* inputs_to_use = (i == 0) ? inputs : prev_activations;
        int inputs_size_to_use = (i == 0) ? input_size : network[i - 1].num_neurons;

        if (prev_activations) {
            free(prev_activations);
        }

        prev_activations = get_activations(network[i].neurons, network[i].num_neurons);

        for (int j = 0; j < network[i].num_neurons; ++j) {
            compute_neuron_output(&network[i].neurons[j], inputs_to_use, inputs_size_to_use);
        }
    }
    if (prev_activations) {
        free(prev_activations);
    }
}


double mean_squared_error(double* targets, double* outputs, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        double error = targets[i] - outputs[i];
        sum += error * error;
    }
    return sum / size;
}


void backpropagate(Neuron* neurons, double* inputs, double* targets, int input_size, int num_neurons, int output_size, double learning_rate) {
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double error = targets[j] - neurons[i].output;
            double delta = learning_rate * error * neurons[i].output * (1.0 - neurons[i].output);
            for (int k = 0; k < input_size; ++k) {
                double weight_update = delta * inputs[k];
                neurons[i].weights[k] += weight_update;
            }
            neurons[i].bias += delta;
        }
    }
}


void backward(Layer* network, double* inputs, double* targets, int num_layers, int input_size, int output_size, double learning_rate) {
    double* next_activations = NULL;
    double* prev_activations = NULL;

    for (int i = num_layers - 1; i > 0; --i) {
        prev_activations = get_activations(network[i - 1].neurons, network[i - 1].num_neurons);

        if (i == num_layers - 1) {
            backpropagate(network[i].neurons, prev_activations, targets, network[i - 1].num_neurons, network[i].num_neurons, output_size, learning_rate);
        } else {
            next_activations = get_activations(network[i + 1].neurons, network[i + 1].num_neurons);
            backpropagate(network[i].neurons, prev_activations, next_activations, network[i - 1].num_neurons, network[i].num_neurons, network[i + 1].num_neurons, learning_rate);
            free(next_activations);
        }
        free(prev_activations);
    }

    next_activations = get_activations(network[1].neurons, network[1].num_neurons);
    backpropagate(network[0].neurons, inputs, next_activations, input_size, network[0].num_neurons, network[1].num_neurons, learning_rate);
    free(next_activations);
}
