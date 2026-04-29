import argparse
import inspect
import time
from argparse import Namespace


def genrecedit_parse_args():
    parser = argparse.ArgumentParser(
        description="GenRecEdit for generative recommendation model editing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("GenRecEdit Model")
    model_group.add_argument("--model_name", type=str, default="TIGER", help="GenRec model name")
    model_group.add_argument("--pretrained_model_path", type=str, default="", help="Path to the pretrained checkpoint")

    data_group = parser.add_argument_group("GenRecEdit Data")
    data_group.add_argument("--covariance_data_file", type=str, default="", help="JSON file for covariance statistics")
    data_group.add_argument("--edit_requests_file", type=str, default="", help="JSON file with GenRecEdit requests")
    data_group.add_argument("--edit_name", type=str, default="", help="Name used in the saved deltaW filename")
    data_group.add_argument("--max_rows", type=float, default=0.0, help="Dataset sampling ratio")
    data_group.add_argument("--cache_dir", type=str, default=None, help="GenRec dataset cache directory")

    output_group = parser.add_argument_group("GenRecEdit Output")
    output_group.add_argument("--output_dir", type=str, default="results", help="Base output directory")
    output_group.add_argument("--category", type=str, default="", help="Amazon category name")
    output_group.add_argument("--log_dir", type=str, default=None, help="GenRec log directory")
    output_group.add_argument("--tensorboard_log_dir", type=str, default=None, help="GenRec tensorboard directory")

    edit_group = parser.add_argument_group("GenRecEdit Hyperparameters")
    edit_group.add_argument(
        "--pos2layer",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Position-to-decoder-layer mapping shared by GenRecEdit and TIGER inference",
    )
    edit_group.add_argument("--cov_lambda", type=int, default=10000, help="Covariance regularization weight")
    edit_group.add_argument("--number_knowledge", type=int, default=5, help="Number of edit knowledge entries")

    cache_group = parser.add_argument_group("GenRecEdit Cache")
    cache_group.add_argument("--covariance_cache_dir", type=str, default="cache/covariance", help="Covariance cache dir")

    return parser.parse_args()


def genrecedit_init_hparams(args):
    from .hparams import GenRecEditHyperParams
    if isinstance(args, Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)

    valid_keys = set(inspect.signature(GenRecEditHyperParams).parameters.keys())
    filtered = {key: value for key, value in args_dict.items() if key in valid_keys}
    return GenRecEditHyperParams(**filtered)


def genrecedit_handle_cache_args(args) -> bool:
    return False


def genrecedit_initialize_model(args):
    from .model_bundle import GenRecEditModelBundle

    mt = GenRecEditModelBundle(args)
    print(f"GenRecEdit model loaded: decoder_layers={mt.num_decoder_layers}, vocab_size={mt.tokenizer.vocab_size}")
    return mt


def genrecedit_run_from_args(args):
    if genrecedit_handle_cache_args(args):
        return

    from .editor import GenRecEdit
    from .io_utils import genrecedit_load_json

    mt = genrecedit_initialize_model(args)
    hparams = genrecedit_init_hparams(args)

    edit_requests = genrecedit_load_json(args.edit_requests_file)
    print(f"Loaded {len(edit_requests)} GenRecEdit requests from {args.edit_requests_file}")

    editor = GenRecEdit(hparams)
    editor.genrecedit_run(mt=mt, edit_targets=edit_requests, args=args)


def genrecedit_main():
    start_time = time.time()
    args = genrecedit_parse_args()
    genrecedit_run_from_args(args)
    elapsed = time.time() - start_time
    print(f"GenRecEdit finished. Total time: {elapsed:.2f}s")
