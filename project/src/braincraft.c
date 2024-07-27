#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "braincraft.h"

NeuralNetwork* create_network(int num_layers, Optimizer optimizer, LossFunction loss_func, float learning_rate, float momentum, float beta1, float beta2) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for NeuralNetwork.\n");
        return NULL;
    }

    nn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    if (nn->layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for layers.\n");
        free(nn);
        return NULL;
    }

    nn->num_layers = num_layers;
    nn->optimizer = optimizer;
    nn->loss_func = loss_func;
    nn->learning_rate = learning_rate;
    nn->momentum = momentum;
    nn->beta1 = beta1;
    nn->beta2 = beta2;

    return nn;
}

void destroy_layer(Layer *layer) {
    if (layer == NULL) return;

    for (int i = 0; i < layer->output_size; ++i) {
        if (layer->neurons[i].weights != NULL) free(layer->neurons[i].weights);
        if (layer->neurons[i].velocity != NULL) free(layer->neurons[i].velocity);
        if (layer->neurons[i].m != NULL) free(layer->neurons[i].m);
        if (layer->neurons[i].v != NULL) free(layer->neurons[i].v);
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
    if (layer->neurons == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for neurons.\n");
        return;
    }

    layer->outputs = (float *)malloc(output_size * sizeof(float));
    if (layer->outputs == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for outputs.\n");
        free(layer->neurons);
        return;
    }

    layer->sums = (float *)malloc(output_size * sizeof(float));
    if (layer->sums == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for sums.\n");
        free(layer->neurons);
        free(layer->outputs);
        return;
    }

    float limit = sqrt(6.0 / (input_size + output_size));

    for (int i = 0; i < output_size; ++i) {
        layer->neurons[i].weights = (float *)malloc(input_size * sizeof(float));
        if (layer->neurons[i].weights == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for neuron weights.\n");
            // Handle memory cleanup for previously allocated neurons
            for (int k = 0; k < i; ++k) {
                free(layer->neurons[k].weights);
            }
            free(layer->neurons);
            free(layer->outputs);
            free(layer->sums);
            return;
        }
        layer->neurons[i].velocity = (float *)calloc(input_size, sizeof(float));
        layer->neurons[i].m = (float *)calloc(input_size, sizeof(float));
        layer->neurons[i].v = (float *)calloc(input_size, sizeof(float));

        for (int j = 0; j < input_size; ++j) {
            layer->neurons[i].weights[j] = (float)rand() / RAND_MAX * 2 * limit - limit;
        }
        layer->neurons[i].bias = (float)rand() / RAND_MAX * 2 * limit - limit;
    }
}

float sum(Neuron *neuron, int input_size, float *inputs) {
    float result = neuron->bias;
    for (int i = 0; i < input_size; ++i) {
        result += neuron->weights[i] * inputs[i];
    }
    return result;
}

void forward_pass(NeuralNetwork *nn, float *inputs) {
    if (nn == NULL || inputs == NULL) return;

    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];
        float *previous_layer_outputs = (i == 0) ? inputs : nn->layers[i - 1].outputs;

        for (int j = 0; j < layer->output_size; ++j) {
            float x = sum(&layer->neurons[j], layer->input_size, previous_layer_outputs);
            layer->sums[j] = x;
            if (layer->activation_func != SOFTMAX) {
                layer->outputs[j] = activation_func(layer->activation_func, x, layer->alpha);
            }
        }

        if (layer->activation_func == SOFTMAX) {
            softmax(layer->sums, layer->outputs, layer->output_size);
        }
    }

    nn->predicts = nn->layers[nn->num_layers - 1].outputs;
}

void backward_pass(NeuralNetwork *nn, float *targets);
