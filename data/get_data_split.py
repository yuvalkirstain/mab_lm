
import argparse
import json
import os

from tqdm import tqdm


def main(args):
    # read wikitext-103 data file
    with open(args.train_data_file, "r", encoding='utf8') as fd:
        train_lines = [line for line in fd.readlines()]
    with open(args.valid_data_file, "r", encoding='utf8') as fd:
        valid_lines = [line for line in fd.readlines()]
    with open(args.test_data_file, "r", encoding='utf8') as fd:
        test_lines = [line for line in fd.readlines()]

    # load wiki info
    with open(args.train_wiki_info, "r", encoding='utf8') as fd:
        train_info = json.load(fd)
    with open(args.valid_wiki_info, "r", encoding='utf8') as fd:
        valid_info = json.load(fd)
    with open(args.test_wiki_info, "r", encoding='utf8') as fd:
        test_info = json.load(fd)

    # create new data split
    for split_lines, split_info, split_name in \
            zip([train_lines, valid_lines, test_lines],
                [train_info, valid_info, test_info],
                ["train", "valid", "test"]):
        in_domain_lines = []
        out_domain_lines = []
        in_domain_page_count = 0
        out_domain_page_count = 0

        for title, info in tqdm(split_info.items()):
            if "what" in info and info["what"] in args.split_what:
                for line_idx in range(info["start_line_idx"], info["end_line_idx"]+1):
                    in_domain_lines.append(split_lines[line_idx])
                in_domain_page_count += 1
            else:
                for line_idx in range(info["start_line_idx"], info["end_line_idx"]+1):
                    out_domain_lines.append(split_lines[line_idx])
                out_domain_page_count += 1

        # write data files
        with open(os.path.join(args.output_dir, f"wiki-{args.split_what}-in-domain.{split_name}.tokens"), "w",
                  encoding='utf8') as fd:
            fd.writelines(in_domain_lines)
        with open(os.path.join(args.output_dir, f"wiki-{args.split_what}-out-domain.{split_name}.tokens"), "w",
                  encoding='utf8') as fd:
            fd.writelines(out_domain_lines)

        print(f"wrote {split_name} data files for '{args.split_what}': "
              f"{len(in_domain_lines)}/{len(out_domain_lines)} in/out-domain lines, "
              f"{in_domain_page_count}/{out_domain_page_count} in/out-domain pages")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split_what', nargs='+',
                        help='value of "what" property based on which the data split will be created.')
    parser.add_argument('--train_wiki_info', type=str, default='raw_data/wikitext-103/wiki.train.tokens.title_map.wiki_info',
                        help='path to validation set wiki info')
    parser.add_argument('--valid_wiki_info', type=str, default='raw_data/wikitext-103/wiki.valid.tokens.title_map.wiki_info',
                        help='path to validation set wiki info')
    parser.add_argument('--test_wiki_info', type=str,
                        default='raw_data/wikitext-103/wiki.test.tokens.title_map.wiki_info',
                        help='path to test set wiki info')
    parser.add_argument('--train_data_file', type=str, default='raw_data/wikitext-103/wiki.train.tokens',
                        help='path to the original training set')
    parser.add_argument('--valid_data_file', type=str, default='raw_data/wikitext-103/wiki.valid.tokens',
                        help='path to the original validation set')
    parser.add_argument('--test_data_file', type=str, default='raw_data/wikitext-103/wiki.test.tokens',
                        help='path to the original test set')
    parser.add_argument('--output_dir', type=str, default='raw_data/wikitext-103',
                        help='path to output dir')
    args = parser.parse_args()

    for arg in vars(args):
        if arg == "split_what":
            continue
        arg_value = getattr(args, arg)
        assert os.path.exists(arg_value), arg_value

    main(args)

