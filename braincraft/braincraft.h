#ifndef _BRAINCRAFT_H
#define _BRAINCRAFT_H

#include <stdlib.h>
#include <math.h>

typedef struct{
    double* weights;
    double bias;
    double output;
} Neuron;

typedef struct{
    Neuron* neurons;
    int num_neurons;
} Layer;


void init_layer(Layer* layer, int size, int input_size);
double sigmoid(double x);
void forward_pass(Layer* network, int num_layers, double* inputs, int input_size);
double mean_squared_error(double* targets, double* outputs, int size);
void backpropagate(Layer* network, double *inputs, double *targets, int input_size, int num_neurons, double learning_rate);


#endif /* _BRAINCRAFT_H */
