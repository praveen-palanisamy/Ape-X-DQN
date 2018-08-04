### PyTorch Implementation of Ape-X (Distributed prioritized experience replay) architecture with DQN learner

- Easy-to-follow implementation with comments indicating the algorithm line as described in the paper

### Running the code

1. Setup a conda env with the necessary python packages. Assuming Anaconda is installed, you can run the following command
from the root of this repository:
`conda env create -f conda_env.yaml -n "apex_dqn_pytorch"`

2. Set the configuration parameters suitable for your hardware in [parameters.json](parameters.json)
At the minimum, you should set the `num_actors` parameter under `"Actor"` and the `"Replay Memory"` 'soft_capacity` based on
the number of CPU cores and RAM available.

3. Launch the training process using the following command:
`python main.py`

You should see episode stats printed out to the console. You can change the learning environment
using the `"name"`parameter value under `"env_conf"` in [parameters.json](parameters.json)


##### To-Dos:

  -  [ ] Compress state/observations using PNG codec before storing in memory and decompress when needed
  -  [ ] Bias correction in prioritized replay using importance sampling
