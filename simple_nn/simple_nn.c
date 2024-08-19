#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "utils.h"

#define INPUT_SIZE 10
#define NUM_CLASSES 2

int main() {
    srand((unsigned int)time(NULL));

    int num_layers = 2;
    NeuralNetwork *nn = create_network(num_layers, ADAM, MSE, 0.9, 0.9, 0.999, NONE, 0.0);

    init_layer(&nn->layers[0], INPUT_SIZE, 4, SIGMOID, 0.0);
    init_layer(&nn->layers[1], 4, NUM_CLASSES, SIGMOID, 0.0);

    print_neural_network(nn);

    float inputs[] = {1, 0, 1, 1, 0, 1, 1, 0, 0, 1};
    float targets[] = {1, 0};

    float learning_rate = 0.01;
    int num_epochs = 1000;

    for (int i = 0; i <= num_epochs; ++i) {
        forward_pass(nn, inputs);
        compute_gradients(nn, inputs, targets);
        update_weights(nn, learning_rate);
        init_zero_gradients(nn);

        if (i % 100 == 0) {
            printf("Epoch: %d\n", i);
            print_vector("Predicts: ", nn->predicts, NUM_CLASSES, 0, 6);
            printf("Loss: %.9f\n\n", loss_func(nn->loss_func, nn->predicts, targets, NUM_CLASSES));
        }
    }

    print_neural_network(nn);
    destroy_network(nn);
    
    return 0;
}
