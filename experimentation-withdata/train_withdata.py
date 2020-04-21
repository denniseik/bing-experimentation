import argparse
import os

# Parse data location as argument
parser = argparse.ArgumentParser("train")
parser.add_argument("--input_data", type=str, help="input data")
args = parser.parse_args()

# Print data mount location
print("Argument 1: %s" % args.input_data)
