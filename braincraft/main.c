#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"


double* get_outputs(NeuralNetwork *network) {
    double* outputs = (double*)calloc(network->layers[network->num_layers - 1].num_neurons, sizeof(double));
    for (int i = 0; i < network->layers[network->num_layers - 1].num_neurons; ++i) {
        outputs[i] = network->layers[network->num_layers - 1].neurons[i].output;
    }
    return outputs;
}


void print_outputs(NeuralNetwork *network) {
    printf("Outputs: ");
    for (int i = 0; i < network->layers[network->num_layers - 1].num_neurons; ++i) {
        printf("%lf ", network->layers[network->num_layers - 1].neurons[i].output);
    }
    printf("\n");
}


int main() {
    srand(time(NULL));

    int num_layers = 3;
    int input_size = 5;
    int hidden_size_1 = 10;
    int output_size = 4;

    double inputs[] = {1.0, 1.2, 0.035, 1.011, -1.7};
    double targets[] = {0.0, 0.1, 0.7, 0.23};

    NeuralNetwork *network = create_network(num_layers);

    init_layer(&network->layers[0], input_size, hidden_size_1, sigmoid, NULL);
    init_layer(&network->layers[1], network->layers[0].num_neurons, hidden_size_1, sigmoid, NULL);
    init_layer(&network->layers[2], network->layers[1].num_neurons, output_size, sigmoid, NULL);

    int num_epochs = 1000;
    double learning_rate = 0.01;

    for (int i = 0; i <= num_epochs; ++i) {
        forward(network, inputs);
        backward(network, inputs, targets, output_size, learning_rate);
        if (i % 100 == 0) {
            printf("Iteration: %d\n", i);
            print_outputs(network);
            double* outputs = get_outputs(network);
            printf("Loss: %lf\n\n", mean_squared_error(targets, outputs, output_size));
            free(outputs);
        }
    }

    print_network_info(network);

    destroy_network(network);

    return 0;
}
