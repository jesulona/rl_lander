# Lunar Lander Reinforcement Learning

This project is a reinforcement learning experiment using the Lunar Lander environment from OpenAI's Gymnasium.

## Requirements

- Python 3.8 or above
- [Gymnasium](https://github.com/Farama-Foundation/gymnasium) for the Lunar Lander environment.
- [PyTorch](https://pytorch.org/) with CUDA 12.1 support for model training using GPU acceleration.
- [Matplotlib](https://matplotlib.org/) for visualization of training results.
- [NumPy](https://numpy.org/) for numerical computations.

## Installation

To set up the project environment, ensure that you have `pip` installed and then run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```
## Running the Training

Once you have installed all required dependencies, you can start the training process. To run the code:

1. Open your terminal or command prompt.
2. Navigate to the topmost folder of this repository where the `main.py` file resides.

    ```bash
    cd path/to/repository
    ```

3. Execute the `main.py` script using Python.

    ```bash
    python3 main.py
    ```

By default, the script will train the agent for 1000 episodes. During the training, you will see the episode number and the obtained reward for each episode displayed in the terminal. This output provides insight into how well the agent is learning from the environment over time.

## Training Results

After the training completes, the script will automatically plot the training results. This plot showcases the learning performance comparison between the Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) agents across all episodes.

The plot helps in visualizing the improvement and stability of the agents' learning process over the episodes. It serves as a graphical representation to compare the effectiveness of DQN and DDQN in the given environment.

## Important Notes

- Ensure that you run the `main.py` script from the repository's root to avoid any issues with relative paths.
- The training process might take a considerable amount of time depending on the complexity of the model and the capabilities of your system.
- For GPU acceleration, ensure that your system has a compatible NVIDIA GPU with the correct version of CUDA installed.

## Contributing

If you wish to contribute to this project, you are welcome to submit pull requests. Before you do, please first discuss the change you wish to make via an issue.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments

- OpenAI Gymnasium for providing the environments for training reinforcement learning agents.
- The Python and PyTorch communities for their comprehensive documentation and support.

---


