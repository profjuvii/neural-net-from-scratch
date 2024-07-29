#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "data_loader.h"

int main() {
    srand((unsigned int)time(NULL));

    int num_layers = 4;

    int input_size = 784;
    int num_classes = 10;

    float learning_rate = 0.01;
    float momentum = 0.9;
    float beta1 = 0.9;
    float beta2 = 0.999;

    NeuralNetwork *nn = create_network(num_layers, ADAM, CROSS_ENTROPY, learning_rate, momentum, beta1, beta2);

    init_layer(&nn->layers[0], input_size, 256, RELU, 0.0);
    init_layer(&nn->layers[1], 256, 128, RELU, 0.0);
    init_layer(&nn->layers[2], 128, 64, RELU, 0.0);
    init_layer(&nn->layers[3], 64, num_classes, SOFTMAX, 0.0);

    int batch_size = 64;
    DataLoader *data_loader = create_data_loader(batch_size);
    
    int epochs = 1000;
    for (int i = 0; i <= epochs; ++i) {
        load(data_loader, num_classes); return 1;

        if (i % 10 == 0) {
            printf("\nIteration: %d\n", i);
        }

        for (int j = 0; j < batch_size; ++j) {
            Vector *vector = &data_loader->vectors[j];
            int label = data_loader->labels[j];

            float *targets = (float *)calloc(num_classes, sizeof(float));
            targets[label] = 1.0;

            forward_pass(nn, vector->features);
            backward_pass(nn, vector->features, targets);

            if (i % 10 == 0) {
                printf("  Image: %d\tLoss: %f\n", j, loss_func(nn->loss_func, nn->predicts, targets, num_classes));
            }

            free(targets);
        }
    }

    destroy_data_loader(data_loader);
    destroy_network(nn);

    return 0;
}
