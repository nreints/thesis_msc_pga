import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    type=float,
)
parser.add_argument(
    "-n",
    type=float,
)
parser.add_argument(
    "-d",
    type=str,
)
args = parser.parse_args()
old = args.o
new = args.n

print((new - old) / old)
