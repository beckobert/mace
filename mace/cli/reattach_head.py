from argparse import ArgumentParser

import torch

from mace.tools.scripts_utils import reattach_foundation_head

def main():
    parser = ArgumentParser()
    # TODO: Add some functionality to potentially add multiple heads from a foundation model.
    # Potentially, this can be done rather sloppy in a for loop, looping over the list of heads.
    # Only problem is how to deal with the foundation models from the pre-multihead era.
    parser.add_argument(
        "--target_device",
        "-d",
        help="target device for model. Defaults to current device of model.",
        required=False,
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="File name of output model. Default to something.",
        required=False,
    )
    parser.add_argument(
        "--model_file",
        "-m",
        help="File of trained model, onto which model gets attached.",
        required=True,
    )
    parser.add_argument(
        "--foundation_model_file",
        "-f",
        help="Foundation model, from which head is taken.",
        required=True,
    )
    args = parser.parse_args()
    
    model = torch.load(args.model_file)
    model_foundation = torch.load(args.foundation_model_file)
    torch.set_default_dtype(next(model.parameters()).dtype)

    if args.target_device is None:
        args.target_device = str(next(model.parameters()).device)
    if args.output_file is None:
        args.output_file = f"{args.model_file}.reattached_head.{args.target_device}"

    model_new = reattach_foundation_head(model, model_foundation)
    if args.target_device != str(next(model_new.parameters()).device):
        model_new.to(args.target_device)
    torch.save(model_new, args.output_file)
    

if __name__ == "__main__":
    main()