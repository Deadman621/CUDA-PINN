# Physics-Informed Neural Networks (PINN)

This project implements Physics-Informed Neural Networks (PINN) from scratch using CUDA. The goal of this project is to leverage neural networks to solve partial differential equations (PDEs) by incorporating physical laws into the training process.

## Project Structure

```
PINN
├── src
│   ├── main.cu         # Entry point for the PINN implementation
│   ├── network.cu      # Neural network architecture implementation
│   └── utils.cu        # Utility functions for data handling and processing
├── include
│   ├── network.h       # Header file for neural network classes and functions
│   └── utils.h         # Header file for utility functions and constants
├── CMakeLists.txt      # CMake configuration file
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PINN
   ```

2. **Install dependencies:**
   Ensure you have CMake and a CUDA-capable compiler installed on your system.

3. **Build the project:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

After building the project, you can run the PINN implementation using the following command:

```bash
./pinn_executable
```

Replace `pinn_executable` with the actual name of the compiled binary.

## Overview

The PINN framework consists of several components:

- **Neural Network Architecture:** Defined in `network.cu` and `network.h`, this includes layers, forward and backward passes, and loss calculations.
- **Utility Functions:** Implemented in `utils.cu` and declared in `utils.h`, these functions assist with data handling and normalization.
- **Main Execution:** The `main.cu` file initializes the network, sets up the training process, and manages CUDA kernel execution.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.