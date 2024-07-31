#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "data_loader.h"
#include "utils.h"

int main() {
    srand((unsigned int)time(NULL));

    int num_layers = 3;

    int input_size = 28 * 28;
    int num_classes = 10;

    float learning_rate = 0.001;
    float momentum = 0.9;
    float beta1 = 0.9;
    float beta2 = 0.999;

    NeuralNetwork *nn = create_network(num_layers, ADAM, CROSS_ENTROPY, learning_rate, momentum, beta1, beta2, NONE, 0.0);

    init_layer(&nn->layers[0], input_size, 128, RELU, 0.0);
    init_layer(&nn->layers[1], 128, 64, RELU, 0.0);
    init_layer(&nn->layers[2], 64, num_classes, SOFTMAX, 0.0);

    int batch_size = 64;
    char *path = "/Users/profjuvi/Datasets/MNIST/";

    DataLoader *data_loader = create_data_loader(batch_size, input_size, path);

    int num_epochs = 1000;

    // Training loop
    for (int i = 0; i <= num_epochs; ++i) {
        if (i % 10 == 0) {
            printf("Iteration: %d\n", i);
        }

        if (load_data(data_loader, num_classes) == -1) {
            fprintf(stderr, "Error: Failed to load data.\n");
            break;
        }

        float total_loss = 0.0;

        for (int j = 0; j < batch_size; ++j) {
            float *features = data_loader->vectors[j];
            float *targets = create_targets(num_classes, data_loader->labels[j]);

            forward_pass(nn, features);
            backward_pass(nn, features, targets);

            double loss = loss_func(nn->loss_func, nn->predicts, targets, num_classes);
            total_loss += loss;
            
            if (i % 10 == 0) {
                printf("  Image: %d\n  Loss: %f\n", j + 1, loss);
                print_vector("    Predicts:\t", nn->predicts, num_classes, 10, 6);
                print_vector("    Targets:\t", targets, num_classes, 10, 0);
                printf("\n");
            }

            free(targets);
        }

        if (i % 10 == 0) {
            printf("Total loss: %f\n\n", total_loss / batch_size);
        }
    }

    destroy_data_loader(data_loader);
    destroy_network(nn);

    return 0;
}
