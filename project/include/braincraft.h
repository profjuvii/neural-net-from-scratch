#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activation_funcs.h"
#include "loss_funcs.h"
#include "optimizers.h"

typedef struct {
    float *weights;
    float bias;
    float *velocity;
    float *m, *v;
} Neuron;

typedef struct {
    int input_size;
    int output_size; // number of neurons
    Neuron *neurons;
    ActivationFunction activation_func;
    float *outputs;
    float *sums;
    float alpha;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    float *predicts;
    Optimizer optimizer;
    LossFunction loss_func;
    float learning_rate;
    float momentum;
    float beta1, beta2;
} NeuralNetwork;

NeuralNetwork* create_network(int num_layers, Optimizer optimizer, LossFunction loss_func, float learning_rate, float momentum, float beta1, float beta2);
void destroy_network(NeuralNetwork *nn);
void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation_func, float alpha);

void forward_pass(NeuralNetwork *nn, float *inputs);
int backward_pass(NeuralNetwork *nn, float *inputs, float *targets);

#endif // BRAINCRAFT_H
