import brax_utils
import predictor
from ppo import train
from private_envs import multipusher, turningant

from brax.io import model
from argparse import ArgumentParser
import wandb
import os

parser = ArgumentParser()
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--horizon", type=int, default=20)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--subsample_batchsize", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=100_000_000)
parser.add_argument("--num_minibatches", type=int, default=16)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_evals", type=int, default=40)
parser.add_argument("--n_truncated_rollouts", type=int, default=128)
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--unroll_length", type=int, default=30)
parser.add_argument("--mi-eps", type=float, default=0.01)
parser.add_argument("--reward_scaling", type=float, default=5.0)
parser.add_argument("--discount", type=float, default=0.95)
parser.add_argument("--k_p", type=float, default=1.0)
parser.add_argument("--k_i", type=float, default=0.1)
parser.add_argument("--k_d", type=float, default=0.001)
parser.add_argument("--target_kl", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=3e-4)

parser.add_argument("--no_save_video", action="store_true")
parser.add_argument("--no_save_model", action="store_true")
parser.add_argument("--no_train_predictor", action="store_true")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--hide_us", action="store_true")
parser.add_argument("--model_path", type=str, default="./params")
parser.add_argument("--ant", action="store_true")
parser.add_argument("--freeze_balls", action="store_true")

args = parser.parse_args()

# Set MI-eps to 10.0 to train an unconstrained model
if args.mi_eps == 10.0:
    print("Unconstrained model is training!")

if args.ant:
    env = turningant.TurningAnt()
else:
    env = multipusher.MultiPusher(hide_us=args.hide_us, freeze_balls=args.freeze_balls)

if args.no_wandb:
    wandb.init(mode="disabled")
else:
    wandb.init(name=args.name, config=args.__dict__)

predictor.SEQUENCE_LENGTH = args.T
predictor.VOCAB_SIZE = env.action_size
predictor.env = env

# TRANSFORMER ARGS
transformer_config = {
    "vocab_size": predictor.VOCAB_SIZE,
    "output_vocab_size": predictor.OUTPUT_VOCAB_SIZE,
    "emb_dim": predictor.EMB_SIZE,
    "num_heads": predictor.NUM_HEADS,
    "qkv_dim": predictor.EMB_SIZE,
    "mlp_dim": predictor.EMB_SIZE,
    "num_layers": predictor.NUM_LAYERS,
    "max_len": predictor.SEQUENCE_LENGTH,
    "kernel_init": predictor.w_init,
    "logits_via_embedding": False,
}

make_inference_fn, params, _ = train.train(
    env,
    num_timesteps=args.timesteps,
    num_evals=args.num_evals,
    reward_scaling=args.reward_scaling,
    episode_length=args.T,
    normalize_observations=True,
    action_repeat=1,
    seed=args.seed,
    unroll_length=args.unroll_length,
    num_minibatches=args.num_minibatches,
    num_updates_per_batch=1,
    discounting=args.discount,
    learning_rate=args.learning_rate,
    entropy_cost=1e-2,
    num_envs=args.num_envs,
    batch_size=args.batchsize,
    get_us_fn=env.get_us,
    horizon=args.horizon,
    n_truncated_rollouts=args.n_truncated_rollouts,
    subsample_batch_size=args.subsample_batchsize,
    pid_parameters=(args.k_p, args.k_i, args.k_d),
    MI_eps=args.mi_eps,
    run_name=args.name,
    transformer_config=transformer_config,
    target_kl=args.target_kl,
    debug=args.debug,
)

if not args.no_save_model and args.name is not None:
    filename = args.name.replace(" ", "")
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    model_path = f"{args.model_path}/{filename}"
    model.save_params(model_path, params)

    if not args.no_train_predictor:
        predictor.full_trajectory_MI(
            model_path=model_path, seed=args.seed + 1, model_config=transformer_config, layers={}
        )

if not args.no_save_video:
    print("Rendering Video...")
    if args.ant:
        hfov = 100
    else:
        hfov = 29.0

    wandb.log(
        {
            "vid_1": brax_utils.make_video(
                params,
                make_inference_fn,
                env,
                args.T,
                flip_camera=not args.ant,
                curr_seed=-1,
                n_seeds=1,
                hfov=hfov,
                use_antialising=True,
                width=800,
                height=600,
            )
        }
    )
    wandb.log(
        {
            "vid_5": brax_utils.make_video(
                params,
                make_inference_fn,
                env,
                args.T,
                flip_camera=not args.ant,
                curr_seed=2,
                n_seeds=5,
                hfov=hfov,
                use_antialising=True,
            )
        }
    )

    print("Finished Video!")
