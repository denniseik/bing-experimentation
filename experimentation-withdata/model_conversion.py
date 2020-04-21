import argparse
import os

# Parse data location as argument
parser = argparse.ArgumentParser("train")
parser.add_argument(
    "--input_data",
    type=str,
    default=".",
    help="input data"
)
parser.add_argument(
    "--output_data",
    type=str,
    default="outputs",
    help="input data"
)
args = parser.parse_args()

# Print data mount location
print("Input data folder: %s" % args.input_data)
print("Output data folder: %s" % args.output_data)

print("My model conversion runner file")