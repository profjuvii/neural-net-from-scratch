#include <stdio.h>
#include <stdlib.h>
#include <synapse.h>
#include <time.h>

int main(void) {
    srand((unsigned int)time(NULL));

    // Data preparation
    int input_size = 28 * 28;
    int num_classes = 10;
    
    char *train_path = "/Users/profjuvi/Datasets/mnist_png/train";
    char *test_path = "/Users/profjuvi/Datasets/mnist_png/test";

    int batch_size = 64;

    DataLoader *dataloader_train = create_dataloader(train_path, batch_size, input_size);
    DataLoader *dataloader_test = create_dataloader(test_path, batch_size, input_size);

    set_mean_std_params(0.5f, 0.5f);

    // Initialization of the neural network
    int num_layers = 3;

    create_network(num_layers);

    init_layer(input_size, 128, ReLU);
    init_layer(128, 64, ReLU);
    init_layer(64, num_classes, Softmax);

    setup_loss_function(CrossEntropy);
    setup_optimizer(Adam, 0.001f);

    // Model training
    int num_epochs = 100;

    float *targets = (float *)calloc(num_classes, sizeof(float));

    for (int epoch = 0; epoch <= num_epochs; ++epoch) {
        get_next_batch(dataloader_train, num_classes);
        float total_loss = 0.0f;

        for (int i = 0; i < batch_size; ++i) {
            float *features = dataloader_train->batch[i].features;
            targets[dataloader_train->batch[i].label] = 1.0f;
        
            forward(features);

            float loss = loss_function(targets);
            total_loss += loss;

            if (auto_epoch(epoch, num_epochs)) {
                if (i == 0) printf("Batch processing:\n");
                printf("  Sample: %d\n  Loss: %.6f\n\n", i + 1, loss);
            }

            compute_gradients(features, targets);

            targets[dataloader_train->batch[i].label] = 0.0f;
        }

        update_weights();
        zero_gradients();

        if (auto_epoch(epoch, num_epochs)) {
            printf("Epoch: %d\nTotal loss: %.6f\n\n", epoch, total_loss / batch_size);
        }
    }

    // Model testing
    get_next_batch(dataloader_test, num_classes);
    float correct_predicts = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        float *features = dataloader_test->batch[i].features;

        int label = dataloader_test->batch[i].label;
        targets[label] = 1.0f;

        forward(features);
        float *predicts = get_network_predictions();
        correct_predicts += (label == index_of_max(predicts, num_classes));

        printf("Number %d: %s\nPredictions: ", label, (label == index_of_max(predicts, num_classes)) ? "True" : "False");
        print_vector(predicts, num_classes);
        printf("\n");

        targets[label] = 0.0f;

        free(predicts);
    }

    float accuracy = (float)correct_predicts / batch_size * 100;
    printf("Accuracy: %.1f%%\n", accuracy);

    if (accuracy >= 90.0f) {
        save_network("../models/mnist_classifier*.bin");
        printf("Model saved successfully.\n");
    }

    // print_network();

    // Free memory
    free(targets);
    destroy_dataloader(dataloader_train);
    destroy_dataloader(dataloader_test);
    destroy_network();

    return 0;
}
