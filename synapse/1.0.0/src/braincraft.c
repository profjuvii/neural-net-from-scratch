#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "braincraft.h"
#include "utils.h"

typedef struct {
    float *weights;
    float *weight_grads;
    float bias;
    float bias_grad;
    void *optimizer_params;
} Neuron;

typedef struct {
    Neuron *neurons;
    float *activations;
    int num_neurons;
    int input_size;
    float *weighted_sums;
    ActivationFunction activation_function;
} Layer;

static int net_num_layers = 0;
static int is_network_initialized = 0;
static Layer *layers = NULL;

static LossFunction net_loss_function = -1;

static float net_learning_rate = 0.01f;
static Optimizer net_optimizer = SGD;

void create_network(const int num_layers) {
    if (num_layers <= 0 || num_layers > 20) {
        fprintf(stderr, "Error: The number of layers must be between 1 and 20.\n");
        exit(EXIT_FAILURE);
    }

    layers = (Layer *)malloc(num_layers * sizeof(Layer));
    if (!layers) {
        fprintf(stderr, "Error: Memory allocation failed for layers.\n");
        exit(EXIT_FAILURE);
    }

    net_num_layers = num_layers;
}

// Box-Muller transform with polar coordinates
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

void init_weights(float *weights, const int size, const ActivationFunction activation_function) {
    float scale = 1.0f;

    if (activation_function == RELU || activation_function == LEAKY_RELU) {
        scale = sqrtf(2.0f / size); // He initialization
    } else {
        scale = sqrtf(1.0f / size); // Xavier initialization
    }

    for (int i = 0; i < size; ++i) {
        weights[i] = random_normal() * scale;
    }
}

void init_layer(const int input_size, const int num_neurons, const ActivationFunction activation_function) {
    static int index = 0;
    if (index == net_num_layers) {
        fprintf(stderr, "Error: Out of neural network layers.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (input_size <= 0 || num_neurons <= 0) {
        fprintf(stderr, "Error: Invalid input size or number of neurons.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (activation_function < LINEAR || activation_function > SOFTMAX) {
        fprintf(stderr, "Error: Invalid activation function.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    Layer *layer = &layers[index];

    layer->num_neurons = num_neurons;
    layer->input_size = input_size;
    layer->activation_function = activation_function;

    layer->neurons = (Neuron *)malloc(num_neurons * sizeof(Neuron));
    layer->activations = (float *)malloc(num_neurons * sizeof(float));
    layer->weighted_sums = (float *)malloc(num_neurons * sizeof(float));

    if (!layer->neurons || !layer->activations || !layer->weighted_sums) {
        fprintf(stderr, "Error: Memory allocation failed for layer components.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_neurons; ++i) {
        Neuron *neuron = &layer->neurons[i];

        neuron->weights = (float *)malloc(input_size * sizeof(float));
        neuron->weight_grads = (float *)calloc(input_size, sizeof(float));
        neuron->optimizer_params = NULL;

        if (!neuron->weights || !neuron->weight_grads) {
            fprintf(stderr, "Error: Memory allocation failed for neuron components.\n");
            destroy_network();
            exit(EXIT_FAILURE);
        }
        
        init_weights(neuron->weights, input_size, activation_function);

        neuron->bias = 0.0f;
        neuron->bias_grad = 0.0f;
    }

    ++index;

    if (index == net_num_layers) {
        is_network_initialized = 1;
    }
}

void destroy_network(void) {
    if (!layers) return;

    for (int l = 0 ; l < net_num_layers; ++l) {
        Layer *layer = &layers[l];
        
        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            if (neuron->weights) free(neuron->weights);
            if (neuron->weight_grads) free(neuron->weight_grads);

            if (neuron->optimizer_params) {
                switch (net_optimizer) {
                    case MOMENTUM:
                    case ADAGRAD:
                    case RMSPROP: {
                        OptParams *params = (OptParams *)neuron->optimizer_params;
                        if (params->data) free(params->data);
                        break;
                    }
                    case ADAM: {
                        AdamParams *params = (AdamParams *)neuron->optimizer_params;
                        if (params->m) free(params->m);
                        if (params->v) free(params->v);
                        break;
                    }
                    default: break;
                }
                free(neuron->optimizer_params);
            }
        }

        if (layer->neurons) free(layer->neurons);
        if (layer->weighted_sums) free(layer->weighted_sums);
        if (layer->activations) free(layer->activations);
    }

    free(layers);
}

void load_network(const char *path) {
    if (is_network_initialized) {
        fprintf(stderr, "Error: Network is initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (!path || strlen(path) == 0) {
        fprintf(stderr, "Error: Invalid file path.\n");
        return;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open file.\n");
        return;
    }

    // Read the number of layers
    fread(&net_num_layers, sizeof(int), 1, fp);

    // Allocate memory for the layers
    layers = (Layer *)malloc(net_num_layers * sizeof(Layer));
    if (!layers) {
        fprintf(stderr, "Error: Memory allocation failed for layers.\n");
        fclose(fp);
        return;
    }

    for (int l = 0; l < net_num_layers; ++l) {
        Layer *layer = &layers[l];

        // Read number of neurons and input size
        fread(&layer->num_neurons, sizeof(int), 1, fp);
        fread(&layer->input_size, sizeof(int), 1, fp);
        fread(&layer->activation_function, sizeof(ActivationFunction), 1, fp);

        // Allocate memory for neurons, activations, and weighted_sums
        layer->neurons = (Neuron *)malloc(layer->num_neurons * sizeof(Neuron));
        layer->activations = (float *)malloc(layer->num_neurons * sizeof(float));
        layer->weighted_sums = (float *)malloc(layer->num_neurons * sizeof(float));

        if (!layer->neurons || !layer->activations || !layer->weighted_sums) {
            fprintf(stderr, "Error: Memory allocation failed for layer components.\n");
            destroy_network();
            exit(EXIT_FAILURE);
            fclose(fp);
            return;
        }

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            // Allocate memory for neuron weights, gradients, and velocity
            neuron->weights = (float *)malloc(layer->input_size * sizeof(float));
            neuron->weight_grads = (float *)calloc(layer->input_size, sizeof(float));
            neuron->optimizer_params = NULL;

            if (!neuron->weights || !neuron->weight_grads) {
                fprintf(stderr, "Error: Memory allocation failed for neuron components.\n");
                destroy_network();
                exit(EXIT_FAILURE);
                fclose(fp);
                return;
            }

            neuron->bias_grad = 0.0f;

            // Read neuron data
            fread(&neuron->bias, sizeof(float), 1, fp);
            fread(neuron->weights, sizeof(float), layer->input_size, fp);
        }
    }

    is_network_initialized = 1;
    fclose(fp);
}

void save_network(const char *path) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (!path || strlen(path) == 0) {
        fprintf(stderr, "Error: Invalid file path.\n");
        return;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open file.\n");
        return;
    }

    // Write the number of layers
    fwrite(&net_num_layers, sizeof(int), 1, fp);

    for (int l = 0; l < net_num_layers; ++l) {
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

void print_neuron(const Neuron neuron, const int input_size, const int idx) {
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

void print_network(void) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    for (int l = 0; l < net_num_layers; ++l) {
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

void setup_loss_function(const LossFunction loss_function) {
    if (loss_function < MSE || loss_function > CROSS_ENTROPY) {
        fprintf(stderr, "Error: Invalid loss function.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    net_loss_function = loss_function;
}

void setup_optimizer(const Optimizer optimizer, const float learning_rate) {
    if (optimizer < SGD || optimizer > ADAM) {
        fprintf(stderr, "Error: Invalid optimizer.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (learning_rate <= 0.0f) {
        fprintf(stderr, "Error: Learning rate must be positive.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    net_optimizer = optimizer;
    net_learning_rate = learning_rate;
    
    if (net_optimizer == SGD) return;

    for (int l = 0; l < net_num_layers; ++l) {
        Layer *layer = &layers[l];

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            switch (net_optimizer) {
                case MOMENTUM:
                case ADAGRAD:
                case RMSPROP: {
                    OptParams *params = (OptParams *)malloc(sizeof(OptParams));
                    if (!params) {
                        fprintf(stderr, "Error: Memory allocation failed for neuron components.\n");
                        destroy_network();
                        exit(EXIT_FAILURE);
                    }

                    params->data = (float *)calloc(layer->input_size, sizeof(float));
                    if (!params->data) {
                        fprintf(stderr, "Error: Memory allocation failed for neuron components.\n");
                        destroy_network();
                        free(params);
                        exit(EXIT_FAILURE);
                    }

                    neuron->optimizer_params = params;
                    break;
                }

                case ADAM: {
                    AdamParams *params = (AdamParams *)malloc(sizeof(AdamParams));
                    if (!params) {
                        fprintf(stderr, "Error: Memory allocation failed for neuron components.\n");
                        destroy_network();
                        exit(EXIT_FAILURE);
                    }

                    params->m = (float *)calloc(layer->input_size, sizeof(float));
                    params->v = (float *)calloc(layer->input_size, sizeof(float));
                    if (!params->m || !params->v) {
                        fprintf(stderr, "Error: Memory allocation failed for neuron components.\n");
                        destroy_network();
                        free(params);
                        exit(EXIT_FAILURE);
                    }

                    neuron->optimizer_params = params;
                    break;
                }

                default: break;
            }
        }
    }
}

float* get_network_predictions(void) {
    Layer *last_layer = &layers[net_num_layers - 1];
    float *predictions = (float *)malloc(last_layer->num_neurons * sizeof(float));
    if (!predictions) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }
    memcpy(predictions, last_layer->activations, last_layer->num_neurons * sizeof(int));
    return predictions;
}

float loss_function(const float *targets) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (net_loss_function < MSE || net_loss_function > CROSS_ENTROPY) {
        fprintf(stderr, "Error: Loss function is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    Layer *last_layer = &layers[net_num_layers - 1];

    switch (net_loss_function) {
        case MSE: return mse(last_layer->activations, targets, last_layer->num_neurons);
        case CROSS_ENTROPY: return cross_entropy(last_layer->activations, targets, last_layer->num_neurons);
    }

    return 0.0f;
}

float weighted_sum(float *weights, const float *inputs, const int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += inputs[i] * weights[i];
    }
    return sum;
}

void forward(const float *inputs) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    for (int l = 0 ; l < net_num_layers; ++l) {
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

            } else if (l < net_num_layers - 1) {
                fprintf(stderr, "Error: Softmax should only be applied to the output layer.\n");
                destroy_network();
                exit(EXIT_FAILURE);
            }
        }
    }

    // Apply softmax to the output layer
    Layer *last_layer = &layers[net_num_layers - 1];
    if (last_layer->activation_function == SOFTMAX) {
        softmax(last_layer->weighted_sums, last_layer->activations, last_layer->num_neurons);
    }
}

void compute_gradients(const float *inputs, const float *targets) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    if (net_loss_function < MSE || net_loss_function > CROSS_ENTROPY) {
        fprintf(stderr, "Error: Loss function is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    Layer *last_layer = &layers[net_num_layers - 1];

    float *gradients = (float *)malloc(layers[0].num_neurons * sizeof(float));
    float *next_gradients = (float *)malloc(layers[0].num_neurons * sizeof(float));
    
    // Calculation for the last layer
    for (int i = 0; i < last_layer->num_neurons; ++i) {
        Neuron *neuron = &last_layer->neurons[i];

        // Calculation of gradients for the last layer
        if (net_loss_function == MSE && last_layer->activation_function != SOFTMAX) {
            float loss = last_layer->activations[i] - targets[i];
            float activation_derivative = activation_function_derivative(last_layer->activation_function, last_layer->weighted_sums[i]);
            gradients[i] = loss * activation_derivative;

        } else if (net_loss_function == CROSS_ENTROPY && last_layer->activation_function == SOFTMAX) {
            gradients[i] = last_layer->activations[i] - targets[i];

        } else {
            fprintf(stderr, "Error: Unsupported loss function or activation function combination.\n");
            destroy_network();
            exit(EXIT_FAILURE);
        }

        // Calculation of gradients of weights
        for (int j = 0; j < last_layer->input_size; ++j) {
            neuron->weight_grads[j] += gradients[i] * layers[net_num_layers - 2].activations[j];
        }

        neuron->bias_grad += gradients[i];
    }
    
    // Backpropagation
    for (int l = net_num_layers - 2; l >= 0; --l) {
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

void update_weights(void) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    static int t = 1;

    for (int l = 0 ; l < net_num_layers; ++l) {
        Layer *layer = &layers[l];
        
        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            optimize(
                net_optimizer,
                neuron->optimizer_params,
                neuron->weights,
                neuron->weight_grads,
                layer->input_size,
                net_learning_rate,
                t
            );

            neuron->bias -= net_learning_rate * neuron->bias_grad;
        }
    }

    ++t;
}

void zero_gradients(void) {
    if (!is_network_initialized) {
        fprintf(stderr, "Error: Network is not initialized.\n");
        destroy_network();
        exit(EXIT_FAILURE);
    }

    for (int l = 0 ; l < net_num_layers; ++l) {
        Layer *layer = &layers[l];

        for (int i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];

            memset(neuron->weight_grads, 0, layer->input_size * sizeof(float));
            neuron->bias_grad = 0.0f;
        }
    }
}
