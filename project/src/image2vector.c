#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include "image2vector.h"

void image2vector(const char *filename, float *buffer, int *size) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file %s.\n", filename);
        return;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png == NULL) {
        fprintf(stderr, "Error: Failed to create PNG read structure.\n");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (info == NULL) {
        fprintf(stderr, "Error: Failed to create PNG info structure.\n");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error during PNG initialization.\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    png_read_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    png_bytep *row_pointers = png_get_rows(png, info);
    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);

    *size = width * height;

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_byte pixel = row[x * 4];
            buffer[y * width + x] = (float)pixel / 255.0;
        }
    }

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
}
