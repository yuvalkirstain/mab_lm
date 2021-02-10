
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
    new_train_lines = []
    new_valid_lines = []
    new_test_lines = []
    train_page_count = 0
    valid_page_count = 0
    test_page_count = 0
    for split_lines, split_info, split_name in \
            zip([train_lines, valid_lines, test_lines],
                [train_info, valid_info, test_info],
                ["train", "valid", "test"]):
        for title, info in tqdm(split_info.items()):
            # dev set includes only in-domain test examples.
            if split_name == "valid" and "what" in info and info["what"] == args.split_what:
                for line_idx in range(info["start_line_idx"], info["end_line_idx"]+1):
                    new_valid_lines.append(split_lines[line_idx])
                valid_page_count += 1
            # test set includes only in-domain test examples.
            elif split_name == "test" and "what" in info and info["what"] == args.split_what:
                for line_idx in range(info["start_line_idx"], info["end_line_idx"]+1):
                    new_test_lines.append(split_lines[line_idx])
                test_page_count += 1

            # train set includes out-of-domain train and dev examples + in-domain train examples.
            elif split_name in ["train", "valid"]:
                for line_idx in range(info["start_line_idx"], info["end_line_idx"]+1):
                    new_train_lines.append(split_lines[line_idx])
                train_page_count += 1

    print(f"created a data split for '{args.split_what}': "
          f"train pages {train_page_count}, valid pages {valid_page_count}, test pages {test_page_count}, "
          f"train lines {len(new_train_lines)}, valid lines {len(new_valid_lines)}, test lines {len(new_test_lines)}.")

    # write data files
    with open(os.path.join(args.output_dir, f"wiki-{args.split_what}.train.tokens"), "w", encoding='utf8') as fd:
        fd.writelines(new_train_lines)
    with open(os.path.join(args.output_dir, f"wiki-{args.split_what}.valid.tokens"), "w", encoding='utf8') as fd:
        fd.writelines(new_valid_lines)
    with open(os.path.join(args.output_dir, f"wiki-{args.split_what}.test.tokens"), "w", encoding='utf8') as fd:
        fd.writelines(new_test_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split_what', type=str, default='',
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

