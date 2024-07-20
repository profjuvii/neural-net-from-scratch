#ifndef _BRAINCRAFT_H
#define _BRAINCRAFT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double* weights;
    double bias;
    double output;
} Neuron;

typedef struct {
    Neuron* neurons;
    int num_neurons;
} Layer;


void init_layer(Layer* layer, int size, int input_size);
void free_network(Layer* network, int num_layers);
double sigmoid(double x);
void forward(Layer* network, int num_layers, double* inputs, int input_size);
double mean_squared_error(double* targets, double* outputs, int size);
void backward(Layer* network, double *inputs, double *targets, int num_layers, int input_size, int output_size, double learning_rate);


#endif /* _BRAINCRAFT_H */
