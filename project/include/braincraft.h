#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activation_funcs.h"
#include "optimizers.h"

typedef enum {
    MSE,
    CROSS_ENTROPY
} LossFunction;

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
    float *velocity;
    float *m, *v;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    float *predictions;
    Optimizer optimizer;
    LossFunction loss_func;
    float learning_rate;
    float momentum;
    float beta1, beta2;
} NeuralNetwork;

NeuralNetwork *create_network(int num_layers, Optimizer optimizer, LossFunction loss_func, float learning_rate, float momentum, float beta1, float beta2);
void destroy_network(NeuralNetwork *nn);
void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation_func, float alpha);

void forward_pass(NeuralNetwork *nn, float *inputs);
void backward_pass(NeuralNetwork *nn, float *targets);
float mse(float *predictions, float *targets, int size);

#endif // BRAINCRAFT_H