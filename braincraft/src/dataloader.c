#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "dataloader.h"
#include "image2vector.h"

void destroy_batch(Sample *batch, int size) {
    if (!batch) return;
    for (int i = 0; i < size; ++i) {
        if (batch[i].features) free(batch[i].features);
    }
    free(batch);
}

void destroy_dataloader(DataLoader *dataloader) {
    if (!dataloader) return;
    destroy_batch(dataloader->batch, dataloader->batch_size);
    free(dataloader);
}

DataLoader* create_dataloader(char* dataset_path, int batch_size, int input_size) {
    DataLoader *dataloader = (DataLoader *)malloc(sizeof(DataLoader));
    if (dataloader == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for dataloader.\n");
        return NULL;
    }

    dataloader->batch = (Sample *)malloc(batch_size * sizeof(Sample));
    if (dataloader == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for batch.\n");
        free(dataloader);
        return NULL;
    }

    for (int i = 0; i < batch_size; ++i) {
        Sample *sample = &dataloader->batch[i];
        sample->features = (float *)malloc(input_size * sizeof(float));

        if (sample->features == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for features.\n");
            destroy_dataloader(dataloader);
            return NULL;
        }
    }

    dataloader->dataset_path = dataset_path;
    dataloader->batch_size = batch_size;
    dataloader->input_size = input_size;

    return dataloader;
}

int get_next_batch(DataLoader *dataloader, int num_classes) {
    int images_per_class = dataloader->batch_size / num_classes;
    int remaining_images = dataloader->batch_size % num_classes;

    int batch_index = 0;

    for (int label = 0; label < num_classes; ++label) {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%d", dataloader->dataset_path, label);

        DIR *dir = opendir(dir_path);
        if (dir == NULL) {
            fprintf(stderr, "Error: Failed to open directory: %s\n", dir_path);
            exit(EXIT_FAILURE);
        }

        struct dirent *entry;
        char **file_names = NULL;
        int file_count = 0;

        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            file_names = realloc(file_names, (file_count + 1) * sizeof(char *));
            file_names[file_count] = strdup(entry->d_name);
            file_count++;
        }
        closedir(dir);

        int num_to_pick = images_per_class + (remaining_images > 0 ? 1 : 0);
        remaining_images = remaining_images > 0 ? remaining_images - 1 : 0;

        for (int i = 0; i < num_to_pick; ++i) {
            int file_index = rand() % file_count;
            char *file_name = file_names[file_index];

            char file_path[512];
            snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, file_name);

            int size = 0;
            image2vector(file_path, dataloader->batch[batch_index].features, &size);
            // printf("%s\n", file_path);
            if (dataloader->input_size != size) {
                free(file_names);
                return -1;
            }

            normalize_vector(dataloader->batch[batch_index].features, dataloader->input_size, 0.5, 0.5);
            dataloader->batch[batch_index].label = label;

            batch_index++;
        }

        for (int j = 0; j < file_count; ++j) {
            free(file_names[j]);
        }
        free(file_names);
    }

    return 0;
}

float* create_targets(int num_classes, int label) {
    float *targets = (float *)calloc(num_classes, sizeof(float));
    targets[label] = 1.0;
    return targets;
}
