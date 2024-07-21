#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"


void print_network(Layer* network, int num_layers, int input_size) {
    for (int i = 0; i < num_layers; ++i) {
        printf("Layer: %02d\n",i + 1);
        printf("---------\n");
        for (int j = 0; j < network[i].num_neurons; ++j) {
            printf("%d neuron:\n\tWeights: ", j + 1);
            if (i == 0) {
                for (int k = 0; k < input_size; ++k) {
                    printf("%lf ", network[i].neurons[j].weights[k]);
                }
            } else {
                for (int k = 0; k < network[i - 1].num_neurons; ++k) {
                    printf("%lf ", network[i].neurons[j].weights[k]);
                }
            }
            printf("\n\tBias: %lf\n\tOutput: %lf\n", network[i].neurons[j].bias, network[i].neurons[j].output);
        }
        printf("\n\n");
    }
}


void print_outputs(Layer* network, int num_layers) {
    printf("Outputs: ");
    for (int i = 0; i < network[num_layers - 1].num_neurons; ++i) {
        printf("%lf ", network[num_layers - 1].neurons[i].output);
    }
    printf("\n");
}


double* get_outputs(Layer* network, int num_layers) {
    double* outputs = (double*)calloc(network[num_layers - 1].num_neurons, sizeof(double));
    for (int i = 0; i < network[num_layers - 1].num_neurons; ++i) {
        outputs[i] = network[num_layers - 1].neurons[i].output;
    }
    return outputs;
}


int main() {
    srand(time(NULL));

    int num_layers = 4;
    int input_size = 6;
    int output_size = 1;
    int num_epochs = 1000;
    double learning_rate = 0.01;

    double inputs[] = {1.0, -0.45, 0.71, 0.3, 1.7, 2.2};
    double targets[] = {0.5};

    Layer* network = (Layer*)calloc(num_layers, sizeof(Layer));

    // initialize all layers
    init_layer(&network[0], 10, input_size); // the first layer
    init_layer(&network[1], 8, network[0].num_neurons); // the second layer
    init_layer(&network[2], 4, network[1].num_neurons); // the third layer
    init_layer(&network[3], output_size, network[2].num_neurons); // the last layer

    // training loop
    for (int i = 0; i <= num_epochs; ++i) {
        forward(network, num_layers, inputs, input_size);
        backward(network, inputs, targets, num_layers, input_size, output_size, learning_rate);
        if (i % 10 == 0) {
            printf("Iteration: %d\n", i);
            print_outputs(network, num_layers);
            double* outputs = get_outputs(network, num_layers);
            printf("Loss: %lf\n\n", mean_squared_error(targets, outputs, output_size));
            free(outputs);
        }
    }

    print_network(network, num_layers, input_size);
    free_network(network, num_layers);

    return 0;
}
