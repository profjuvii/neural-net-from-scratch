#include <time.h>
#include "braincraft.h"


void print_outputs(Layer* network, int num_layers) {
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


void free_network(Layer* network, int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        for (int j = 0; j < network[i].num_neurons; ++j) {
            free(network[i].neurons[j].weights);
        }
        free(network[i].neurons);
    }
    free(network);
}


int main() {
    // srand(time(NULL));
    int num_layers = 3;
    int input_size = 4;
    int num_epochs = 10;

    Layer* network = (Layer*)calloc(num_layers, sizeof(Layer));

    // initialize all layers
    init_layer(&network[0], 6, input_size); // first layer
    init_layer(&network[1], 4, network[0].num_neurons); // second layer
    init_layer(&network[2], 2, network[1].num_neurons); // third layer

    double inputs[] = {1.2, -0,45, 0.71, 0.3};
    double targets[] = {1, 0};

    for (int i = 0; i < num_epochs; ++i) {
        forward(network, num_layers, inputs, input_size);
        backward(network, inputs, targets, num_layers, input_size, 0.1);
    }

    print_outputs(network, num_layers);
    double* outputs = get_outputs(network, num_layers);
    printf("Loss: %lf\n\n", mean_squared_error(targets, outputs, network[2].num_neurons));

    free(outputs);
    free_network(network, num_layers);

    return 0;
}
