#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "braincraft.h"

NeuralNetwork* create_network(
    int num_layers,
    Optimizer optimizer,
    LossFunction loss_func,
    float momentum,
    float beta1,
    float beta2,
    Regularization reg,
    float reg_param
) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for NeuralNetwork.\n");
        return NULL;
    }

    nn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    if (nn->layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for layers.\n");
        free(nn);
        return NULL;
    }

    nn->num_layers = num_layers;
    nn->optimizer = optimizer;
    nn->loss_func = loss_func;
    nn->momentum = momentum;
    nn->beta1 = beta1;
    nn->beta2 = beta2;
    nn->reg = reg;
    nn->reg_param = reg_param;

    return nn;
}

void destroy_layer(Layer *layer) {
    if (layer == NULL) return;

    for (int i = 0; i < layer->output_size; ++i) {
        if (layer->neurons[i].weights != NULL) free(layer->neurons[i].weights);
        if (layer->neurons[i].weight_grads != NULL) free(layer->neurons[i].weight_grads);
        if (layer->neurons[i].velocity != NULL) free(layer->neurons[i].velocity);
        if (layer->neurons[i].m != NULL) free(layer->neurons[i].m);
        if (layer->neurons[i].v != NULL) free(layer->neurons[i].v);
    }
    
    if (layer->neurons != NULL) free(layer->neurons);
    if (layer->activations != NULL) free(layer->activations);
    if (layer->sums != NULL) free(layer->sums);
}

void destroy_network(NeuralNetwork *nn) {
    if (nn == NULL) return;

    for (int i = 0; i < nn->num_layers; ++i) {
        destroy_layer(&nn->layers[i]);
    }

    free(nn->layers);
    free(nn);
}

void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation_func, float alpha) {
    if (layer == NULL) return;

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_func = activation_func;
    layer->alpha = alpha;

    layer->neurons = (Neuron *)malloc(output_size * sizeof(Neuron));
    if (layer->neurons == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for neurons.\n");
        return;
    }

    layer->activations = (float *)malloc(output_size * sizeof(float));
    if (layer->activations == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for outputs.\n");
        free(layer->neurons);
        return;
    }

    layer->sums = (float *)malloc(output_size * sizeof(float));
    if (layer->sums == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for sums.\n");
        free(layer->neurons);
        free(layer->activations);
        return;
    }

    float limit = sqrt(6.0 / (input_size + output_size));

    for (int i = 0; i < output_size; ++i) {
        layer->neurons[i].weights = (float *)malloc(input_size * sizeof(float));
        if (layer->neurons[i].weights == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for neuron weights.\n");
            // Handle memory cleanup for previously allocated neurons
            for (int k = 0; k < i; ++k) {
                free(layer->neurons[k].weights);
            }
            free(layer->neurons);
            free(layer->activations);
            free(layer->sums);
            return;
        }
        layer->neurons[i].weight_grads = (float *)malloc(input_size * sizeof(float));
        layer->neurons[i].velocity = (float *)malloc(input_size * sizeof(float));
        layer->neurons[i].m = (float *)malloc(input_size * sizeof(float));
        layer->neurons[i].v = (float *)malloc(input_size * sizeof(float));

        for (int j = 0; j < input_size; ++j) {
            layer->neurons[i].weights[j] = (float)rand() / RAND_MAX * 2 * limit - limit;
        }
        layer->neurons[i].bias = (float)rand() / RAND_MAX * 2 * limit - limit;
    }
}

float sum(Neuron *neuron, int input_size, float *inputs) {
    float result = neuron->bias;
    for (int i = 0; i < input_size; ++i) {
        result += neuron->weights[i] * inputs[i];
    }
    return result;
}

void forward_pass(NeuralNetwork *nn, float *inputs) {
    if (nn == NULL || inputs == NULL) return;

    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];
        float *previous_layer_outputs = (i == 0) ? inputs : nn->layers[i - 1].activations;

        for (int j = 0; j < layer->output_size; ++j) {
            float x = sum(&layer->neurons[j], layer->input_size, previous_layer_outputs);
            layer->sums[j] = x;
            if (layer->activation_func != SOFTMAX) {
                layer->activations[j] = activation_func(layer->activation_func, x, layer->alpha);
            }
        }

        if (layer->activation_func == SOFTMAX) {
            softmax(layer->sums, layer->activations, layer->output_size);
        }
    }

    nn->predicts = nn->layers[nn->num_layers - 1].activations;
}

float compute_weighted_sum_delta(Layer *layer, float grad, int idx) {
    float sum = 0.0;
    for (int i = 0; i < layer->output_size; ++i) {
        sum += layer->neurons[i].weights[idx] * grad;
    }
    return sum;
}

float* copy_vector(float *vector, int size) {
    float *copy = (float *)malloc(size * sizeof(float));
    if (copy == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for vector copy.\n");
        return NULL;
    }

    for (int i = 0; i < size; ++i) {
        copy[i] = vector[i];
    }

    return copy;
}

void l1_reg(Layer *layer, float *losses, float lambda) {
    for (int i = 0; i < layer->output_size; ++i) {
        Neuron *neuron = &layer->neurons[i];
        float sum = 0.0;

        for (int j = 0; j < layer->input_size; ++j) {
            sum += fabs(neuron->weights[j]);
        }

        losses[i] += lambda * sum;
    }
}

void l2_reg(Layer *layer, float *losses, float lambda) {
    for (int i = 0; i < layer->output_size; ++i) {
        Neuron *neuron = &layer->neurons[i];
        float sum = 0.0;

        for (int j = 0; j < layer->input_size; ++j) {
            sum += neuron->weights[j] * neuron->weights[j];
        }

        losses[i] += lambda * sum;
    }
}

void loss_regularization(Regularization reg, Layer *layer, float *losses, float lambda) {
    switch (reg) {
        case L1:
            l1_reg(layer, losses, lambda);
            break;
        case L2:
            l2_reg(layer, losses, lambda);
            break;
        default:
            return;
    }
}

void compute_gradients(NeuralNetwork *nn, float *inputs, float *targets) {
    Layer *output_layer = &nn->layers[nn->num_layers - 1];

    float *loss_grads = (float *)malloc(output_layer->output_size * sizeof(float));
    if (loss_grads == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for loss gradients.\n");
        return;
    }

    if (nn->loss_func == CROSS_ENTROPY && output_layer->activation_func == SOFTMAX) {
        // Compute the losses in the last layer
        for (int i = 0; i < output_layer->output_size; ++i) {
            loss_grads[i] = output_layer->activations[i] - targets[i];
        }
        
        Layer *prev_layer = &nn->layers[nn->num_layers - 2];
        
        // Update the weights and biases in the last layer
        for (int i = 0; i < output_layer->output_size; ++i) {
            Neuron *neuron = &output_layer->neurons[i];

            for (int j = 0; j < prev_layer->output_size; ++j) {
                neuron->weight_grads[j] += loss_grads[i] * prev_layer->activations[j];
            }

            neuron->bias_grad += loss_grads[i];
        }

    } else if (nn->loss_func == MSE) {
        // Compute the losses in the last layer
        for (int i = 0; i < output_layer->output_size; ++i) {
            loss_grads[i] = 2 * (output_layer->activations[i] - targets[i]) / output_layer->output_size;
        }

        Layer *prev_layer = &nn->layers[nn->num_layers - 2];
        
        // Update the weights and biases in the last layer
        for (int i = 0; i < output_layer->output_size; ++i) {
            Neuron *neuron = &output_layer->neurons[i];
            float delta = loss_grads[i] * activation_func_grad(output_layer->activation_func, output_layer->activations[i], output_layer->alpha);

            for (int j = 0; j < prev_layer->output_size; ++j) {
                neuron->weight_grads[j] += delta * prev_layer->activations[j];
            }

            neuron->bias_grad += delta;
        }
    }

    float *next_grads = copy_vector(loss_grads, output_layer->output_size);
    free(loss_grads);

    for (int l = nn->num_layers - 2; l >= 0; --l) {
        Layer *layer = &nn->layers[l];
        Layer *next_layer = &nn->layers[l + 1];
        Layer *prev_layer = l > 0 ? &nn->layers[l - 1] : NULL;

        float *grads = (float *)calloc(layer->output_size, sizeof(float));
        if (grads == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for gradients.\n");
            free(next_grads);
            return;
        }

        // Compute the gradients of the current layer
        for (int i = 0; i < layer->output_size; ++i) {
            for (int j = 0; j < next_layer->output_size; ++j) {
                float weighted_sum_delta = compute_weighted_sum_delta(next_layer, next_grads[j], j);
                grads[i] = weighted_sum_delta * activation_func_grad(layer->activation_func, layer->sums[i], layer->alpha);
            }
        }

        // Compute the weights and biases gradients in the current layer
        for (int i = 0; i < layer->output_size; ++i) {
            Neuron *neuron = &layer->neurons[i];

            // Compute the weight gradients of the current layer
            for (int j = 0; j < layer->input_size; ++j) {
                neuron->weight_grads[j] += grads[i] * (l > 0 ? prev_layer->activations[j] : inputs[j]);
            }

            neuron->bias_grad += grads[i];
        }

        free(next_grads);
        next_grads = copy_vector(grads, layer->output_size);

        free(grads);
    }

    free(next_grads);
}

void weight_regularization(Regularization reg, float weight, float *grad_weight, float lambda) {
    switch (reg) {
        case L1: 
            if (weight != 0) {
                *grad_weight += lambda * (weight > 0 ? 1 : -1);
            }
            break;
        case L2:
            *grad_weight += 2 * lambda * weight;
            break;
        default:
            return;
    }
}

void update_weights(NeuralNetwork *nn, float learning_rate) {
    for (int i = nn->num_layers - 1; i >= 0; --i) {
        Layer *layer = &nn->layers[i];

        for (int j = 0; j < layer->output_size; ++j) {
            Neuron *neuron = &layer->neurons[j];
            
            switch (nn->optimizer) {
                case SGD:
                    sgd(neuron->weights, neuron->weight_grads, layer->input_size, learning_rate);
                    break;
                case MOMENTUM:
                    momentum(neuron->weights, neuron->weight_grads, neuron->velocity, layer->input_size, learning_rate, nn->momentum);
                    break;
                case ADAM:
                    adam(neuron->weights, neuron->weight_grads, neuron->m, neuron->v, layer->input_size, learning_rate, nn->beta1, nn->beta2, 1e-8);
                    break;
                default:
                    break;
            }

            neuron->bias -= learning_rate * neuron->bias_grad;
        }
    }
}

void init_zero_gradients(NeuralNetwork *nn) {
    for (int l = 0; l < nn->num_layers; ++l) {
        Layer *layer = &nn->layers[l];

        for (int i = 0; i < layer->output_size; ++i) {
            Neuron *neuron = &layer->neurons[i];

            memset(neuron->weight_grads, 0, layer->input_size * sizeof(float));
            neuron->bias_grad = 0;
        }
    }
}

int save_network(NeuralNetwork *nn, const char *path, const char *model_name) {
    if (nn == NULL) {
        fprintf(stderr, "Error: Failed to save neural network. Network is NULL.\n");
        return 1;
    }

    char file_path[256];
    if ((path && strlen(path) > 0) && (model_name && strlen(model_name) > 0)) {
        sprintf(file_path, "%s%s.bin", path, model_name);
    } else if (!path && (model_name && strlen(model_name) > 0)) {
        sprintf(file_path, "%s.bin", model_name);
    } else if ((path && strlen(path) > 0) && !model_name) {
        sprintf(file_path, "%smodel.bin", path);
    } else if (!path && !model_name) {
        sprintf(file_path, "model.bin");
    }

    FILE *fp = fopen(file_path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file for writing.\n");
        return 1;
    }

    fwrite(&nn->num_layers, sizeof(int), 1, fp);
    fwrite(&nn->optimizer, sizeof(Optimizer), 1, fp);
    fwrite(&nn->loss_func, sizeof(LossFunction), 1, fp);
    fwrite(&nn->momentum, sizeof(float), 1, fp);
    fwrite(&nn->beta1, sizeof(float), 1, fp);
    fwrite(&nn->beta2, sizeof(float), 1, fp);
    fwrite(&nn->reg, sizeof(Regularization), 1, fp);
    fwrite(&nn->reg_param, sizeof(float), 1, fp);

    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];
        fwrite(&layer->input_size, sizeof(int), 1, fp);
        fwrite(&layer->output_size, sizeof(int), 1, fp);
        fwrite(&layer->activation_func, sizeof(ActivationFunction), 1, fp);
        fwrite(&layer->alpha, sizeof(float), 1, fp);

        for (int j = 0; j < layer->output_size; ++j) {
            Neuron *neuron = &layer->neurons[j];
            fwrite(neuron->weights, sizeof(float), layer->input_size, fp);
            fwrite(&neuron->bias, sizeof(float), 1, fp);
            fwrite(neuron->velocity, sizeof(float), layer->input_size, fp);
            fwrite(neuron->m, sizeof(float), layer->input_size, fp);
            fwrite(neuron->v, sizeof(float), layer->input_size, fp);
        }
    }

    fclose(fp);
    return 0;
}

NeuralNetwork* load_network(const char* path) {
    if (path == NULL || strlen(path) == 0) {
        fprintf(stderr, "Error: Failed to load neural network. Path is empty.\n");
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file for reading.\n");
        return NULL;
    }

    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        fclose(fp);
        return NULL;
    }

    fread(&nn->num_layers, sizeof(int), 1, fp);
    fread(&nn->optimizer, sizeof(Optimizer), 1, fp);
    fread(&nn->loss_func, sizeof(LossFunction), 1, fp);
    fread(&nn->momentum, sizeof(float), 1, fp);
    fread(&nn->beta1, sizeof(float), 1, fp);
    fread(&nn->beta2, sizeof(float), 1, fp);
    fread(&nn->reg, sizeof(Regularization), 1, fp);
    fread(&nn->reg_param, sizeof(float), 1, fp);

    nn->layers = (Layer *)malloc(nn->num_layers * sizeof(Layer));
    if (nn->layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for layers.\n");
        free(nn);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];
        fread(&layer->input_size, sizeof(int), 1, fp);
        fread(&layer->output_size, sizeof(int), 1, fp);
        fread(&layer->activation_func, sizeof(ActivationFunction), 1, fp);
        fread(&layer->alpha, sizeof(float), 1, fp);

        layer->neurons = (Neuron *)malloc(layer->output_size * sizeof(Neuron));
        if (layer->neurons == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for neurons.\n");
            destroy_network(nn);
            fclose(fp);
            return NULL;
        }

        layer->sums = (float *)malloc(layer->output_size * sizeof(float));
        if (layer->sums == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for sums.\n");
            destroy_network(nn);
            fclose(fp);
            return NULL;
        }

        layer->activations = (float *)malloc(layer->output_size * sizeof(float));
        if (layer->activations == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for activations.\n");
            destroy_network(nn);
            fclose(fp);
            return NULL;
        }

        for (int j = 0; j < layer->output_size; ++j) {
            Neuron *neuron = &layer->neurons[j];

            neuron->weights = (float *)malloc(layer->input_size * sizeof(float));
            if (neuron->weights == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for neuron weights.\n");
                destroy_network(nn);
                fclose(fp);
                return NULL;
            }

            neuron->weight_grads = (float *)malloc(layer->input_size * sizeof(float));
            if (neuron->weights == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for neuron weight gradients.\n");
                destroy_network(nn);
                fclose(fp);
                return NULL;
            }

            fread(neuron->weights, sizeof(float), layer->input_size, fp);
            fread(&neuron->bias, sizeof(float), 1, fp);

            neuron->velocity = (float *)malloc(layer->input_size * sizeof(float));
            neuron->m = (float *)malloc(layer->input_size * sizeof(float));
            neuron->v = (float *)malloc(layer->input_size * sizeof(float));
            if (!neuron->velocity || !neuron->m || !neuron->v) {
                fprintf(stderr, "Error: Failed to allocate memory for optimizer parameters.\n");
                destroy_network(nn);
                fclose(fp);
                return NULL;
            }
            fread(neuron->velocity, sizeof(float), layer->input_size, fp);
            fread(neuron->m, sizeof(float), layer->input_size, fp);
            fread(neuron->v, sizeof(float), layer->input_size, fp);
        }
    }

    fclose(fp);
    return nn;
}
