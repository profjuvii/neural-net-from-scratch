#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "data_loader.h"

void print_vector(char *text, float *vector, int size) {
    printf("%s", text);
    for (int i = 0; i < size; ++i) {
        printf("%*f ", 4, vector[i]);
    }
    printf("\n");
}

int main() {
    srand((unsigned int)time(NULL));

    int num_layers = 4;

    int input_size = 784;
    int num_classes = 10;

    float learning_rate = 0.001;
    float momentum = 0.9;
    float beta1 = 0.9;
    float beta2 = 0.999;

    NeuralNetwork *nn = create_network(num_layers, ADAM, CROSS_ENTROPY, learning_rate, momentum, beta1, beta2);

    init_layer(&nn->layers[0], input_size, 256, RELU, 0.0);
    init_layer(&nn->layers[1], 256, 128, RELU, 0.0);
    init_layer(&nn->layers[2], 128, 64, RELU, 0.0);
    init_layer(&nn->layers[3], 64, num_classes, SOFTMAX, 0.0);

    int batch_size = 64;
    char *path = "/Users/profjuvi/Datasets/MNIST/";

    DataLoader *data_loader = create_data_loader(batch_size, input_size, path);

    int num_epochs = 1000;

    // Training loop
    for (int i = 0; i <= num_epochs; ++i) {
        if (i % 10 == 0) {
            printf("\nIteration: %d\n", i);
        }

        load_data(data_loader, num_classes);

        for (int j = 0; j < batch_size; ++j) {
            float *features = data_loader->vectors[j];
            float *targets = create_targets(num_classes, data_loader->labels[j]);

            forward_pass(nn, features);
            backward_pass(nn, features, targets);

            if (i % 10 == 0) {
                printf("  Image: %d\n  Loss: %f\n", j + 1, loss_func(nn->loss_func, nn->predicts, targets, num_classes));
                print_vector("    Predicts:\t", nn->predicts, num_classes);
                print_vector("    Targets:\t", targets, num_classes);
                printf("\n");
            }

            free(targets);
        }
    }

    destroy_data_loader(data_loader);
    destroy_network(nn);

    return 0;
}
