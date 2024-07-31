#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activation_funcs.h"
#include "loss_funcs.h"
#include "optimizers.h"

typedef enum {
    NONE,
    L1,     // L1 regularization (Lasso)
    L2,     // L2 regularization (Ridge)
} Regularization;

typedef struct {
    float *weights;
    float bias;
    float *velocity;
    float *m, *v;
} Neuron;

typedef struct {
    int input_size;
    int output_size; // Number of neurons in this layer
    Neuron *neurons;
    ActivationFunction activation_func;
    float *outputs;
    float *sums;
    float alpha; // Parameter for Leaky ReLU activation function
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    float *predicts;
    Optimizer optimizer;
    LossFunction loss_func;
    float learning_rate;
    float momentum;     // Momentum term for optimization (used if applicable)
    float beta1, beta2; // Hyperparameters for Adam optimizer (beta1: first moment, beta2: second moment)
    Regularization reg;
    float reg_param;
} NeuralNetwork;


NeuralNetwork* create_network(
    int num_layers,
    Optimizer optimizer,
    LossFunction loss_func,
    float learning_rate,
    float momentum,
    float beta1,
    float beta2,
    Regularization reg,
    float reg_param
);

void destroy_network(NeuralNetwork *nn);
void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation_func, float alpha);

void forward_pass(NeuralNetwork *nn, float *inputs);
int backward_pass(NeuralNetwork *nn, float *inputs, float *targets);

int save_network(NeuralNetwork *nn, const char *path, const char *model_name);
NeuralNetwork* load_network(const char* path);

#endif // BRAINCRAFT_H
