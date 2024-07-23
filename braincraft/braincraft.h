#ifndef _BRAINCRAFT_H
#define _BRAINCRAFT_H

#include <math.h>

typedef struct {
    double* weights;
    double bias;
    double output;
    double* weight_gradients;
    double bias_gradient;
} Neuron;

typedef double (*ActivationFunc)(double, void *);

typedef struct {
    Neuron *neurons;
    int num_neurons;
    int input_size;
    ActivationFunc func;
    void *params;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} NeuralNetwork;


NeuralNetwork *create_network(int num_layers);
void destroy_network(NeuralNetwork *network);
void init_layer(Layer *layer, int input_size, int num_neurons, ActivationFunc func, void *params);
void print_network_info(const NeuralNetwork *network);


double relu(double x, void *params);
double leaky_relu(double x, void *params);
double elu(double x, void *params);
double selu(double x, void *params);
double hyperbolic_tangent(double x, void *params);
double sigmoid(double x, void *params);
double softmax(double x, void *params);
double swish(double x, void *params);


void forward(NeuralNetwork *network, double *inputs);
void backward(NeuralNetwork *network, double* inputs, double* targets, int output_size, double learning_rate);
double mean_squared_error(double* targets, double* outputs, int size);

#endif /* _BRAINCRAFT_H */
