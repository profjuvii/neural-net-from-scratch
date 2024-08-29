#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activators.h"
#include "losses.h"
#include "optimizers.h"

void create_network(const int num_layers);
void init_layer(const int input_size, const int num_neurons, const ActivationFunction activation_function);
void destroy_network(void);
void load_network(char *path);
void save_network(char *path);
void print_network(void);

void setup_loss_function(const LossFunction loss_function);
void setup_optimizer(const Optimizer optimizer, const float learning_rate);

float* get_network_predictions(void);
float loss_function(const float *targets);

void forward(const float *inputs);
void compute_gradients(const float *inputs, const float *targets);
void update_weights(void);
void zero_gradients(void);

#endif
