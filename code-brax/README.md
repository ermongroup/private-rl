## Privacy-Constrained Reinforcement Learning

Install the dependencies via Conda
````
conda env create -f env.yml
conda activate private_rl_brax
````

To run the Brax experiments, you will need a GPU with cuda 11 enabled. We provide a jupyter notebook - ``brax_envs/VisualizeBraxExps.ipynb`` - that makes visualizing trained multi-pusher and turning-ant models easy. We also have WandB logging enabled by default.

* To run the Multi-Pusher experiments
````
python -u main.py --name "pusher-T100-noreg--5000m-s0" --timesteps 5_000_000_000 --mi-eps 10.0 --seed 0
python -u main.py --name "pusher-T100-reg-h10-mi1e-2-5000m-s0" --timesteps 5_000_000_000 --mi-eps 0.01 --horizon 10 --unroll_length 30 --seed 0
````
* To run the Turning-Ant experiments
````
python -u main.py --name "ant-T100-noreg--1000m-s0" --timesteps 1_000_000_000 --mi-eps 10.0 --seed 0 --ant
python -u main.py --name "ant-T100-reg-h10-mi1e-6-2500m-s0" --timesteps 2_500_000_000 --mi-eps 0.06 --horizon 10 --unroll_length 30 --seed 0 --ant
````

Add the flag ``--mi-eps 10.0`` to train unconstrained models. All of the default hyperparameters are optimized for episode length T=100. We suggest training for 2.5 to 5.0 Billion timesteps for Pusher models and 1.0 to 2.5 Billion timesteps for Ant models. The  horizon (for the truncated loss) should be between 10 and 20 and the unroll-length at 30.

If you are running cpu-based JAX, you will have to reduce the number of timesteps dramatically (e.g. to 5 million).
