#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activators.h"
#include "losses.h"
#include "optimizers.h"

typedef struct {
    float *weights;
    float *weight_grads;
    float bias;
    float bias_grad;
    float *velocity;
} Neuron;

typedef struct {
    Neuron *neurons;
    float *activations;
    int num_neurons;
    int input_size;
    float *weighted_sums;
    ActivationFunction activation_function;
} Layer;

Layer *create_network(int num_layers);
void destroy_network(Layer *layers, int num_layers);
Layer *load_network(int *num_layers, char *path);
void save_network(Layer *layers, int num_layers, char *path);
void print_network(Layer *layers, int num_layers);

void init_layer(Layer *layer, int input_size, int num_neurons, ActivationFunction activation_function);
void forward(Layer *layers, float *inputs, int num_layers);
void compute_gradients(Layer *layers, float *inputs, float *targets, int num_layers, LossFunction loss_function);
void update_weights(Layer *layers, int num_layers, float learning_rate, Optimizer optimizer);
void zero_gradients(Layer *layers, int num_layers);

#endif
