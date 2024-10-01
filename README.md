# ARC-AGI Solver

This project implements a solver for the Abstraction and Reasoning Corpus (ARC) challenge using a HyperNetwork and DynamicMainNetwork architecture.

## Project Overview

The ARC-AGI Solver uses a meta-learning approach to solve ARC tasks. It consists of two main components:

1. A HyperNetwork that generates parameters for the MainNetwork based on input-output examples.
2. A DynamicMainNetwork that uses the generated parameters to solve new instances of the task.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/ARC-AGI-solver.git
   cd ARC-AGI-solver
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train and evaluate the ARC solver, run the main.py script:

```
python main.py --data_dir path/to/arc/data --num_epochs 20 --batch_size 64
```

You can adjust the hyperparameters using command-line arguments. For a full list of available options, run:

```
python main.py --help
```

## File Structure

- `main.py`: The entry point of the program, handles training and evaluation.
- `data/arc_dataset.py`: Implements the ARCDataset class for loading and processing ARC tasks.
- `models/hypernetwork.py`: Implements the HyperNetwork model.
- `models/main_network.py`: Implements the DynamicMainNetwork model.
- `utils/train.py`: Contains the training loop and utility functions.

## Approach

This solver uses a meta-learning approach to tackle ARC tasks:

1. The HyperNetwork takes input-output examples from a task and generates parameters for the MainNetwork.
2. The DynamicMainNetwork uses these generated parameters to process new inputs for the same task.
3. During training, the HyperNetwork learns to generate appropriate parameters for various tasks.
4. During evaluation, the trained HyperNetwork can quickly adapt the MainNetwork to new, unseen tasks.

This approach allows the model to adapt to new tasks without requiring retraining, making it suitable for the diverse range of problems in the ARC dataset.

## Contributing

Contributions to improve the ARC-AGI Solver are welcome. Please feel free to submit issues and pull requests.

## License

This project is open-source and available under the MIT License.