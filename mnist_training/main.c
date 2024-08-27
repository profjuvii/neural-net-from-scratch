#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "dataloader.h"
#include "utils.h"

int main() {
    srand((unsigned int)time(NULL));

    // Data preparation
    int input_size = 28 * 28;
    int num_classes = 10;
    
    char *train_path = "/Users/profjuvi/Datasets/mnist_png/train";
    char *test_path = "/Users/profjuvi/Datasets/mnist_png/test";

    int batch_size = 64;
    DataLoader *dataloader_train = create_dataloader(train_path, batch_size, input_size);
    DataLoader *dataloader_test = create_dataloader(test_path, batch_size, input_size);

    // Initialization of the neural network
    int num_layers = 3;

    Layer *nn = create_network(num_layers);
    Layer *last_layer = &nn[num_layers - 1];

    init_layer(&nn[0], input_size, 128, RELU);
    init_layer(&nn[1], 128, 64, RELU);
    init_layer(&nn[2], 64, num_classes, SOFTMAX);

    float *predicts = last_layer->activations;
    LossFunction loss_func = CROSS_ENTROPY; 

    // Model training
    float learning_rate = 0.0008f;
    int num_epochs = 100;

    float *targets = (float *)calloc(num_classes, sizeof(float));

    for (int epoch = 0; epoch <= num_epochs; ++epoch) {
        get_next_batch(dataloader_train, num_classes, 1);
        float total_loss = 0.0f;

        for (int i = 0; i < batch_size; ++i) {
            float *features = dataloader_train->batch[i].features;
            targets[dataloader_train->batch[i].label] = 1.0f;
        
            forward(nn, features, num_layers);

            float loss = loss_function(loss_func, predicts, targets, num_classes);
            total_loss += loss;

            if (auto_epoch(epoch, num_epochs)) {
                if (i == 0) printf("Batch processing:\n");
                printf("  Sample: %d\n  Loss: %.6f\n\n", i + 1, loss);
            }

            compute_gradients(nn, features, targets, num_layers, loss_func);
            update_weights(nn, num_layers, learning_rate, MOMENTUM);
            zero_gradients(nn, num_layers);

            targets[dataloader_train->batch[i].label] = 0.0f;
        }

        if (auto_epoch(epoch, num_epochs)) {
            printf("Epoch: %d\nTotal loss: %.6f\n\n", epoch, total_loss / batch_size);
        }
    }

    // Model testing
    get_next_batch(dataloader_test, num_classes, 1);
    float correct_predicts = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        float *features = dataloader_test->batch[i].features;

        int label = dataloader_test->batch[i].label;
        targets[label] = 1.0f;

        forward(nn, features, num_layers);
        correct_predicts += (label == index_of_max(predicts, num_classes));

        printf("Number %d: %s\nPredictions: ", label, (label == index_of_max(predicts, num_classes)) ? "True" : "False");
        print_vector(predicts, num_classes);
        printf("\n");

        targets[label] = 0.0f;
    }

    float accuracy = (float)correct_predicts / batch_size * 100;
    printf("Accuracy: %.1f%%\n", accuracy);

    // if (accuracy >= 85.0f) {
    //     save_network(nn, num_layers, "/Users/profjuvi/c-projects/braincraft/models/mnist_classifier*.bin");
    //     printf("Model saved successfully.\n");
    // }

    // print_network(nn, num_layers);

    // Free memory
    free(targets);
    destroy_dataloader(dataloader_train);
    destroy_dataloader(dataloader_test);
    destroy_network(nn, num_layers);

    return 0;
}
