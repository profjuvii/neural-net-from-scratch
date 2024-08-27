# Neural Network in the C language

## Project Description
This project features a library for building various architectures of multilayer perceptrons (MLPs) and includes an example of using this library to train a model for digit recognition using the MNIST dataset.

## Project Structure
The project consists of the following main components:

- **`synapse/`**: A library for constructing different architectures of multilayer perceptrons.
  - **`include/`**: Header files for the library.
  - **`lib/`**: Compiled library `libsynapse.a`.
  - **`src/`**: Source files for the library.

- **`mnist_training/`**: Module for training the model based on the MNIST dataset.
  - **`Makefile`**: Build file for the training module.
  - **`main.c`**: Main code for training the model.

- **`mnist_classifier/`**: Module for digit recognition using the trained model.
  - **`Makefile`**: Build file for the classifier.
  - **`images/`**: Test images of digits from 0 to 9.
  - **`main.c`**: Main code for digit recognition.

- **`models/`**: Saved trained model `mnist_classifier.bin`.

## Requirements
To compile and run the project, you need:

- **C Compiler** (e.g., GCC)
- **Make** for automation of the build process

## Usage Instructions

### Building the Library
Navigate to the `synapse` directory and run:

```bash
make
```

This will generate the compiled library `libsynapse.a` in the `lib/` directory.

### Training the Model
Navigate to the `mnist_training` directory and run:

```bash
make
```

This will produce an executable for training the model. Start the training process with:

```bash
./training
```

The training process will save the trained model to `models/mnist_classifier.bin`.

### Digit Recognition
Navigate to the `mnist_classifier` directory and run:

```bash
make
```

This will create an executable for digit recognition. Run the recognition process with:

```bash
./classifier
```

This process will use the saved model to recognize digits from images in the `images/` directory.

## License
The project is licensed under the terms specified in the `LICENSE` file.

## Contact
If you have any questions or suggestions, please contact me at [profjuvi@gmail.com](mailto:profjuvi@gmail.com).