#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "braincraft.h"
#include "utils.h"

Layer *create_network(int num_layers) {
    return (Layer *)malloc(num_layers * sizeof(Layer));
}

void destroy_network(Layer *layers, int num_layers) {
    if (!layers) return;

    for (int l = 0 ; l < num_layers; ++l) {
        Layer *layer = &layers[l];
        
        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            if (neuron->weights) free(neuron->weights);
            if (neuron->weight_grads) free(neuron->weight_grads);
            if (neuron->velocity) free(neuron->velocity);
        }

        if (layer->neurons) free(layer->neurons);
        if (layer->weighted_sums) free(layer->weighted_sums);
        if (layer->activations) free(layer->activations);
    }

    free(layers);
}

Layer *load_network(int *num_layers, char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return NULL;
    }

    // Read the number of layers
    fread(num_layers, sizeof(int), 1, fp);

    // Allocate memory for the layers
    Layer *nn = (Layer *)malloc(*num_layers * sizeof(Layer));
    if (!nn) {
        fprintf(stderr, "Failed to allocate memory for layers.\n");
        fclose(fp);
        return NULL;
    }

    for (int l = 0; l < *num_layers; ++l) {
        Layer *layer = &nn[l];

        // Read number of neurons and input size
        fread(&layer->num_neurons, sizeof(int), 1, fp);
        fread(&layer->input_size, sizeof(int), 1, fp);
        fread(&layer->activation_function, sizeof(ActivationFunction), 1, fp);

        // Allocate memory for neurons, activations, and weighted_sums
        layer->neurons = (Neuron *)malloc(layer->num_neurons * sizeof(Neuron));
        layer->activations = (float *)malloc(layer->num_neurons * sizeof(float));
        layer->weighted_sums = (float *)malloc(layer->num_neurons * sizeof(float));

        if (!layer->neurons || !layer->activations || !layer->weighted_sums) {
            fprintf(stderr, "Failed to allocate memory for layer components.\n");
            fclose(fp);
            return NULL; // Handle memory allocation failure properly
        }

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            // Allocate memory for neuron weights, gradients, and velocity
            neuron->weights = (float *)malloc(layer->input_size * sizeof(float));
            neuron->weight_grads = (float *)calloc(layer->input_size, sizeof(float));
            neuron->velocity = (float *)calloc(layer->input_size, sizeof(float));

            if (!neuron->weights || !neuron->weight_grads || !neuron->velocity) {
                fprintf(stderr, "Failed to allocate memory for neuron components.\n");
                fclose(fp);
                return NULL; // Handle memory allocation failure properly
            }

            neuron->bias_grad = 0.0f;

            // Read neuron data
            fread(&neuron->bias, sizeof(float), 1, fp);
            fread(neuron->weights, sizeof(float), layer->input_size, fp);
        }
    }

    fclose(fp);
    return nn;
}

void save_network(Layer *layers, int num_layers, char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    // Write the number of layers
    fwrite(&num_layers, sizeof(int), 1, fp);

    for (int l = 0; l < num_layers; ++l) {
        Layer *layer = &layers[l];

        // Write number of neurons and input size
        fwrite(&layer->num_neurons, sizeof(int), 1, fp);
        fwrite(&layer->input_size, sizeof(int), 1, fp);
        fwrite(&layer->activation_function, sizeof(ActivationFunction), 1, fp);

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            // Write neuron data
            fwrite(&neuron->bias, sizeof(float), 1, fp);
            fwrite(neuron->weights, sizeof(float), layer->input_size, fp);
        }
    }

    fclose(fp);
}

void print_neuron(Neuron neuron, int input_size, int idx) {
    int width = 24;
    printf("    Neuron %d:\n", idx + 1);

    // Print a neuron weights and bias
    printf("%*s", width, "Weights: ");
    print_vector(neuron.weights, input_size);
    printf("%*s%.6f\n\n", width, "Bias: ", neuron.bias);

    // Print a neuron gradients
    printf("%*s", width, "Weight gradients: ");
    print_vector(neuron.weight_grads, input_size);
    printf("%*s%.6f\n\n", width, "Bias gradient: ", neuron.bias_grad);
}

void print_network(Layer *layers, int num_layers) {
    for (int l = 0; l < num_layers; ++l) {
        Layer *layer = &layers[l];
        printf("-- Layer %d ----------------\n\n", l + 1);
        
        if (layer->num_neurons <= 10) {
            for (int i = 0; i < layer->num_neurons; ++i) {
                print_neuron(layer->neurons[i], layer->input_size, i);
            }
        } else {
            for (int i = 0; i < 5; ++i) {
                print_neuron(layer->neurons[i], layer->input_size, i);
            }
            printf("    ...\n\n");
            for (int i = layer->num_neurons - 5; i < layer->num_neurons; ++i) {
                print_neuron(layer->neurons[i], layer->input_size, i);
            }
        }
        
        // Print a layer activations
        printf("    Activations: ");
        print_vector(layer->activations, layer->num_neurons);
        printf("\n");
    }
}

// Method Box-Muller transform with polar coordinates for generating normal random numbers 
float random_normal() {
    static int flag = 0;
    static float spare;

    if (flag) {
        flag = 0;
        return spare;
    }

    float u1, u2, s;

    do {
        u1 = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        u2 = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        s = u1 * u1 + u2 * u2;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = u2 * s;
    flag = 1;

    return u1 * s;
}

void init_weights(float *weights, int size, ActivationFunction activation_function) {
    float scale = 1.0f;

    if (activation_function == RELU || activation_function == LEAKY_RELU) {
        // He initialization
        scale = sqrtf(2.0f / size);
    } else {
        // Xavier initialization
        scale = sqrtf(1.0f / size);
    }

    for (int i = 0; i < size; ++i) {
        weights[i] = random_normal() * scale;
    }
}

void init_layer(Layer *layer, int input_size, int num_neurons, ActivationFunction activation_function) {
    layer->num_neurons = num_neurons;
    layer->input_size = input_size;
    layer->activation_function = activation_function;

    layer->neurons = (Neuron *)malloc(num_neurons * sizeof(Neuron));
    layer->activations = (float *)malloc(num_neurons * sizeof(float));
    layer->weighted_sums = (float *)malloc(num_neurons * sizeof(float));

    for (int i = 0; i < num_neurons; ++i) {
        Neuron *neuron = &layer->neurons[i];

        neuron->weights = (float *)malloc(input_size * sizeof(float));
        neuron->velocity = (float *)calloc(input_size, sizeof(float));
        neuron->weight_grads = (float *)calloc(input_size, sizeof(float));
        
        // Initialization of neuron weights and bias
        init_weights(layer->neurons[i].weights, input_size, activation_function);

        // Initialization of neuron bias and bias gradient
        neuron->bias = 0.0f;
        neuron->bias_grad = 0.0f;
    }
}

float weighted_sum(float *weights, float *inputs, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += inputs[i] * weights[i];
    }
    return sum;
}

void forward(Layer *layers, float *inputs, int num_layers) {
    for (int l = 0 ; l < num_layers; ++l) {
        Layer *layer = &layers[l];
        
        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            // Weighted sum calculation
            float w_sum = weighted_sum(neuron->weights, (l == 0) ? inputs : layers[l - 1].activations, layer->input_size);
            w_sum += neuron->bias;
            layer->weighted_sums[i] = w_sum;

            // Activation function application
            if (layer->activation_function != SOFTMAX) {
                layer->activations[i] = activation_function(layer->activation_function, w_sum);

            } else if (l < num_layers - 1) {
                destroy_network(layers, num_layers);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Apply softmax to the output layer
    Layer *last_layer = &layers[num_layers - 1];
    if (last_layer->activation_function == SOFTMAX) {
        softmax(last_layer->weighted_sums, last_layer->activations, last_layer->num_neurons);
    }
}

void compute_gradients(Layer *layers, float *inputs, float *targets, int num_layers, LossFunction loss_function) {
    Layer *last_layer = &layers[num_layers - 1];

    float *gradients = (float *)malloc(layers[0].num_neurons * sizeof(float));
    float *next_gradients = (float *)malloc(layers[0].num_neurons * sizeof(float));
    
    // Calculation for the last layer
    for (int i = 0; i < last_layer->num_neurons; ++i) {
        Neuron *neuron = &last_layer->neurons[i];

        // Calculation of gradients for the last layer
        if (loss_function == MSE && last_layer->activation_function != SOFTMAX) {
            float loss = last_layer->activations[i] - targets[i];
            float activation_derivative = activation_function_derivative(last_layer->activation_function, last_layer->weighted_sums[i]);
            gradients[i] = loss * activation_derivative;

        } else if (loss_function == CROSS_ENTROPY && last_layer->activation_function == SOFTMAX) {
            gradients[i] = last_layer->activations[i] - targets[i];

        } else {
            destroy_network(layers, num_layers);
            exit(EXIT_FAILURE);
        }

        // Calculation of gradients of weights
        for (int j = 0; j < last_layer->input_size; ++j) {
            neuron->weight_grads[j] += gradients[i] * layers[num_layers - 2].activations[j];
        }

        neuron->bias_grad += gradients[i];
    }

    // Backpropagation
    for (int l = num_layers - 2; l >= 0; --l) {
        Layer *layer = &layers[l];
        Layer *next_layer = &layers[l + 1];

        float *temp_gradients = gradients;
        gradients = next_gradients;
        next_gradients = temp_gradients;

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            // Calculation of gradients for the hidden layers
            float transposed_weight_gradient = 0.0f;

            for (int j = 0; j < next_layer->num_neurons; ++j) {
                transposed_weight_gradient += next_layer->neurons[j].weights[i] * next_gradients[j];
            }

            float activation_derivative = activation_function_derivative(layer->activation_function, layer->weighted_sums[i]);
            gradients[i] = transposed_weight_gradient * activation_derivative;

            // Calculation of gradients of weights
            for (int j = 0; j < layer->input_size; ++j) {
                neuron->weight_grads[j] += gradients[i] * ((l > 0) ? layers[l + 1].activations[j] : inputs[j]);
            }

            neuron->bias_grad += gradients[i];
        }
    }

    free(gradients);
    free(next_gradients);
}

void update_weights(Layer *layers, int num_layers, float learning_rate, Optimizer optimizer) {
    for (int l = 0 ; l < num_layers; ++l) {
        Layer *layer = &layers[l];
        
        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            apply_optimizer(optimizer, neuron->weights, neuron->weight_grads, neuron->velocity, layer->input_size, learning_rate);
            neuron->bias -= learning_rate * neuron->bias_grad;
        }
    }
}

void zero_gradients(Layer *layers, int num_layers) {
    for (int l = 0 ; l < num_layers; ++l) {
        Layer *layer = &layers[l];

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            memset(neuron->weight_grads, 0, layer->input_size * sizeof(float));
            neuron->bias_grad = 0.0f;
        }
    }
}
