#include <stdio.h>
#include <stdlib.h>
#include "braincraft.h"

NeuralNetwork *create_network(int num_layers, OptimizerType optimizer, float learning_rate) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    nn->num_layers = num_layers;
    nn->optimizer = optimizer;
    nn->learning_rate = learning_rate;
    return nn;
}

void destroy_layer(Layer *layer) {
    if (layer == NULL) return;
    for (int i = 0; i < layer->output_size; ++i) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer->outputs);
    free(layer->sums);
}

void destroy_network(NeuralNetwork *nn) {
    if (nn == NULL) return;
    for (int i = 0; i < nn->num_layers; ++i) {
        destroy_layer(&nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation_func, float alpha) {
    if (layer == NULL) return;

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_func = activation_func;
    layer->alpha = alpha;

    layer->neurons = (Neuron *)malloc(output_size * sizeof(Neuron));
    for (int i = 0; i < output_size; ++i) {
        layer->neurons[i].weights = (float *)malloc(input_size * sizeof(float));
        for (int j = 0; j < input_size; ++j) {
            layer->neurons[i].weights[j] = (float)rand() / RAND_MAX;
        }
        layer->neurons[i].bias = (float)rand() / RAND_MAX;
    }

    layer->outputs = (float *)malloc(output_size * sizeof(float));
    layer->sums = (float *)malloc(output_size * sizeof(float));
}

float sum(Neuron *neuron, int input_size, float *inputs) {
    float res = 0.0;
    for (int i = 0; i < input_size; ++i) {
        res += neuron->weights[i] * inputs[i];
    }
    return res + neuron->bias;
}

float func(ActivationFunction func, float x, float alpha) {
    switch (func) {
        case RELU:
            return relu(x);
        case LEAKY_RELU:
            return leaky_relu(x, alpha);
        case SIGMOID:
            return sigmoid(x);
        case TANH:
            return tanh_activation(x);
        case SOFTMAX:
            return x;
    }
    return x;
}

void forward_pass(NeuralNetwork *nn, float *inputs) {
    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];

        for (int j = 0; j < layer->output_size; ++j) {
            float x = sum(&layer->neurons[j], layer->input_size, i == 0 ? inputs : nn->layers[i - 1].outputs);
            layer->sums[j] = x;
            if (layer->activation_func != SOFTMAX) {
                layer->outputs[j] = func(layer->activation_func, x, layer->alpha);
            }
        }

        if (layer->activation_func == SOFTMAX) {
            softmax(layer->sums, layer->outputs, layer->output_size);
        }
    }
}
