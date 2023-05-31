"""Run finetuning/inference of HuggingFace LLM (seq2seq or CausalLM) for document data extraction."""

import argparse
import data_extraction as pred
import finetuning as tune


def parser():
    """Parse arguments."""
    # "./finetuned" "bigscience/mt0-large" "bigscience/bloomz-1b7"
    parser = argparse.ArgumentParser(
        description="""Run finetuning/inference of HuggingFace LLM (seq2seq or CausalLM) for document data extraction."""
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model (remote or local).",
        default="bigscience/mt0-large"
    )
    #parser.add_argument(
    #    "--action", 
    #    type=str, 
    #    help="What type of action to do: train/inference.",
    #    default="inference"
    #)
    subparsers = parser.add_subparsers(dest='command', help='Different options for different actions.')
    parser_pred_dev = subparsers.add_parser('inference_dev', help='For inference.')
    parser_pred_prod = subparsers.add_parser('inference_prod', help='For inference.')
    parser_tune = subparsers.add_parser('train', help='For finetuning.')
    parser_pred_dev.add_argument(
        "--model_type",
        type=str,
        help="Model type for inference (qa or lm).",
        default="qa"
    )
    parser_pred_prod.add_argument(
        "--model_type",
        type=str,
        help="Model type for inference (qa or lm).",
        default="qa"
    )
    parser_pred_prod.add_argument(
        "--doc_path",
        type=str,
        help="Path to a document for prod inference.",
        required=True
    )
    parser_pred_prod.add_argument(
        "--path_to_save",
        type=str,
        help="Path to where save JSON file.",
        default="./output.json"
    )
    parser_tune.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset for finetuning.",
        default="./data_finetuning"
    )
    parser_tune.add_argument(
        "--train_outputs",
        type=str,
        help="Where to save outputs during/after training.",
        default=["temp", "./temp_finetuned"],
        nargs=2
    )
    parser_tune.add_argument(
        "--bs",
        type=int,
        help="Specify batch size.",
        default=8,
    )
    parser_tune.add_argument(
        "--epochs",
        type=int,
        help="Specify number of epochs.",
        default=3,
    )
    parser_tune.add_argument(
        "--lr",
        type=float,
        help="Specify learning rate.",
        default=5e-6,
    )
    args = parser.parse_args()
    return args
 

def main(args):
    """Run specific functionality depending on the command."""
    if args.command == "train":
        tune.finetune(args.model, args.path_dataset, args.train_outputs, args.bs, args.epochs, args.lr)
    elif args.command == "inference_prod":
        pred.extract_data_from_file(args.model, args.model_type, args.doc_path, args.path_to_save)
    elif args.command == "inference_dev":
        pred.extract_data_from_file_example(args.model, args.model_type)
    else:
        raise ValueError("Action not recognized! Choose between train/inference.")


if __name__ == "__main__":
    args = parser()
    main(args)
