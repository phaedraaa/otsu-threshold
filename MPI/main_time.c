#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GRAY_LEVELS 256

void calculate_histogram(unsigned char *image, int size, int *histogram) {
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}

int otsu_threshold(int *histogram, int total_pixels) {
    int sum = 0, sumB = 0, q1 = 0, q2 = 0;
    float max_var = 0.0, threshold = 0.0;

    for (int i = 0; i < GRAY_LEVELS; i++) {
        sum += i * histogram[i];
    }

    for (int i = 0; i < GRAY_LEVELS; i++) {
        q1 += histogram[i];
        if (q1 == 0) continue;

        q2 = total_pixels - q1;
        if (q2 == 0) break;

        sumB += i * histogram[i];
        float m1 = (float)sumB / q1;
        float m2 = (float)(sum - sumB) / q2;
        float var_between = (float)q1 * q2 * (m1 - m2) * (m1 - m2);

        if (var_between > max_var) {
            max_var = var_between;
            threshold = i;
        }
    }

    return (int)threshold;
}

void apply_threshold(unsigned char *image, int size, int threshold) {
    for (int i = 0; i < size; i++) {
        image[i] = (image[i] > threshold) ? 255 : 0;
    }
}

void read_rawimage(char *fname, unsigned long height, unsigned long width, unsigned char **image) {
    FILE *file = fopen(fname, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(EXIT_FAILURE);
    }
    for (unsigned long i = 0; i < height; i++) {
        fread(image[i], 1, width, file);
    }
    fclose(file);
}

void write_rawimage(char *fname, unsigned long height, unsigned long width, unsigned char **image) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(EXIT_FAILURE);
    }
    for (unsigned long i = 0; i < height; i++) {
        fwrite(image[i], 1, width, file);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    int rank, size;
    unsigned long i;
    double io_start, io_end, scatter_start, scatter_end, compute_start, compute_end, gather_start, gather_end;
    double total_start, total_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    total_start = MPI_Wtime(); // Χρόνος έναρξης συνολικής εκτέλεσης

    unsigned char **image = NULL;
    unsigned long height, width, total_pixels;
    unsigned char *local_image = NULL;
    int *global_histogram = NULL, *local_histogram = NULL;

    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s input_image height width output_image\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char *input_file = argv[1];
    height = atoi(argv[2]);
    width = atoi(argv[3]);
    char *output_file = argv[4];
    total_pixels = height * width;

    if (rank == 0) {
        io_start = MPI_Wtime(); // Χρόνος έναρξης I/O
        image = (unsigned char **)malloc(height * sizeof(unsigned char *));
        for (i = 0; i < height; i++) {
            image[i] = (unsigned char *)malloc(width * sizeof(unsigned char));
        }
        read_rawimage(input_file, height, width, image);
        io_end = MPI_Wtime(); // Χρόνος λήξης I/O
    }

    int local_size = total_pixels / size;
    local_image = (unsigned char *)malloc(local_size * sizeof(unsigned char));

    scatter_start = MPI_Wtime(); // Χρόνος έναρξης Scatter
    if (rank == 0) {
        unsigned char *flat_image = (unsigned char *)malloc(total_pixels * sizeof(unsigned char));
        for (i = 0; i < height; i++) {
            memcpy(flat_image + i * width, image[i], width);
        }
        MPI_Scatter(flat_image, local_size, MPI_UNSIGNED_CHAR, local_image, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        free(flat_image);
    } else {
        MPI_Scatter(NULL, local_size, MPI_UNSIGNED_CHAR, local_image, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
    scatter_end = MPI_Wtime(); // Χρόνος λήξης Scatter

    compute_start = MPI_Wtime(); // Χρόνος έναρξης Υπολογισμού
    local_histogram = (int *)calloc(GRAY_LEVELS, sizeof(int));
    calculate_histogram(local_image, local_size, local_histogram);

    if (rank == 0) global_histogram = (int *)calloc(GRAY_LEVELS, sizeof(int));
    MPI_Reduce(local_histogram, global_histogram, GRAY_LEVELS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int threshold = 0;
    if (rank == 0) {
        threshold = otsu_threshold(global_histogram, total_pixels);
    }
    MPI_Bcast(&threshold, 1, MPI_INT, 0, MPI_COMM_WORLD);
    apply_threshold(local_image, local_size, threshold);
    compute_end = MPI_Wtime(); // Χρόνος λήξης Υπολογισμού

    gather_start = MPI_Wtime(); // Χρόνος έναρξης Gather
    if (rank == 0) {
        unsigned char *flat_image = (unsigned char *)malloc(total_pixels * sizeof(unsigned char));
        MPI_Gather(local_image, local_size, MPI_UNSIGNED_CHAR, flat_image, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        for (i = 0; i < height; i++) {
            memcpy(image[i], flat_image + i * width, width);
        }
        free(flat_image);
        write_rawimage(output_file, height, width, image);
        for (i = 0; i < height; i++) free(image[i]);
        free(image);
        free(global_histogram);
    } else {
        MPI_Gather(local_image, local_size, MPI_UNSIGNED_CHAR, NULL, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
    gather_end = MPI_Wtime(); // Χρόνος λήξης Gather

    free(local_image);
    free(local_histogram);

    total_end = MPI_Wtime(); // Χρόνος λήξης συνολικής εκτέλεσης

    if (rank == 0) {
        printf("I/O Time: %.6f seconds\n", io_end - io_start);
        printf("Scatter Time: %.6f seconds\n", scatter_end - scatter_start);
        printf("Compute Time: %.6f seconds\n", compute_end - compute_start);
        printf("Gather Time: %.6f seconds\n", gather_end - gather_start);
        printf("Total Execution Time: %.6f seconds\n", total_end - total_start);
    }

    MPI_Finalize();
    return 0;
}

