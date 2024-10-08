#include <stdio.h>
#include <stdlib.h>
#include <png.h>

void image2vector(const char *path, float *features) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file %s.\n", path);
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
    png_byte color_type = png_get_color_type(png, info);

    switch (color_type) {
        case PNG_COLOR_TYPE_GRAY:
            for (int y = 0; y < height; ++y) {
                png_bytep row = row_pointers[y];

                for (int x = 0; x < width; ++x) {
                    png_byte px = row[x];
                    features[y * width + x] = (float)px / 255.0f;
                }
            }
            break;
            
        case PNG_COLOR_TYPE_RGB:
            for (int y = 0; y < height; ++y) {
                png_bytep row = row_pointers[y];

                for (int x = 0; x < width; ++x) {
                    png_bytep px = &(row[x * 3]);

                    float r = (float)px[0] / 255.0f;
                    float g = (float)px[1] / 255.0f;
                    float b = (float)px[2] / 255.0f;

                    float pixel_value = 0.299f * r + 0.587f * g + 0.114f * b;
                    features[y * width + x] = pixel_value;
                }
            }
            break;

        case PNG_COLOR_TYPE_RGBA:
            for (int y = 0; y < height; ++y) {
                png_bytep row = row_pointers[y];

                for (int x = 0; x < width; ++x) {
                    png_bytep px = &(row[x * 4]);

                    float r = (float)px[0] / 255.0f;
                    float g = (float)px[1] / 255.0f;
                    float b = (float)px[2] / 255.0f;
                    
                    float pixel_value = 0.299f * r + 0.587f * g + 0.114f * b;
                    features[y * width + x] = pixel_value;
                }
            }
            break;

        default:
            fprintf(stderr, "Unsupported color type: %d\n", color_type);
            png_destroy_read_struct(&png, &info, NULL);
            fclose(fp);
            return;
    }

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
}
