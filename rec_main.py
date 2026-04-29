import argparse

from genrec.utils import parse_command_line_args, get_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TIGER', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2023', help='Dataset name')
    # parser.add_argument('--config_file', type=str, default="genrec/models/TIGER/config.yaml", help='Path to config file')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args) # {} 不用定义直接写
    # import ipdb; ipdb.set_trace()
    pipeline = get_pipeline(args.model)(
        model_name=args.model,
        dataset_name=args.dataset,
        config_dict=command_line_configs,
        # config_file = args.config_file
    )
    # import ipdb; ipdb.set_trace()
    pipeline.run()
