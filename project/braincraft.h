#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activation_funcs.h"

typedef enum {
    SGD,
    MOMENTUM,
    ADAM
} OptimizerType;

typedef struct {
    float *weights;
    float bias;
} Neuron;

typedef struct {
    int input_size;
    int output_size;
    Neuron *neurons;
    ActivationFunction activation_func;
    float *outputs;
    float *sums;
    float alpha;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    OptimizerType optimizer;
    float learning_rate;
} NeuralNetwork;

NeuralNetwork *create_network(int num_layers, OptimizerType optimizer, float learning_rate);
void destroy_network(NeuralNetwork *nn);
void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation, float alpha);
void forward_pass(NeuralNetwork *nn, float *inputs);

#endif // BRAINCRAFT_H
