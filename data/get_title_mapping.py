
import argparse
import json
import os
from tqdm import tqdm


def main(args):
    # read wikitext-103 data file
    with open(args.data_file, "r") as fd:
        lines = [line.strip('\n') for line in fd.readlines()]

    # initialization
    results = {}
    pbar = tqdm(total=len(lines))
    prev_line_idx = 0

    # handle the first title
    title = lines[1][3:-3]
    pbar.update(2)
    line_idx = 2

    # now go over the rest of the file
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith(" = ") and not line.startswith(" = ="):
            results[title] = {"start_line_idx": prev_line_idx, "end_line_idx": line_idx-1}
            prev_line_idx = line_idx
            title = line[3:-3]

        line_idx += 1
        pbar.update(1)

    results[title] = {"start_line_idx": prev_line_idx, "end_line_idx": line_idx - 1}

    # dump results
    with open(args.output_file, "w") as fd:
        json.dump(results, fd, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, default='', help='path to wikitext-103 data file')
    parser.add_argument('output_file', type=str, default='', help='path to output file (json)')
    args = parser.parse_args()

    assert os.path.exists(args.data_file)

    main(args)

