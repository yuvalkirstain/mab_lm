
import argparse
import json
import os
import wptools

from multiprocessing import Pool
from time import sleep


skip_list = ['assessments', 'backlinks', 'claims', 'exhtml', 'exrest', 'extext', 'files',
             'html', 'image', 'iwlinks', 'languages', 'lead', 'links', 'parsetree',
             'random', 'redirects', 'url_raw', 'wikidata_url', 'wikitext', 'backlinks',
             'views', 'contributors', 'watchers', 'infobox', 'extract',
             'length', 'modified', 'imageinfo']
progress_freq = 10


def get_title_info(title_idx, title, timeout, to_sleep, silent):
    if title_idx % progress_freq == 0:
        print(f"processed: {title_idx}", end='\r', flush=True)

    page = wptools.page(title, skip=skip_list, silent=silent)
    page.get(timeout=timeout)
    page.get_more(timeout=timeout)

    page_data = {
        key: page.data[key] for key in page.data
        if key not in skip_list
    }
    page_data["query_title"] = title

    if to_sleep:
        sleep(5)

    return page_data


def query_titles(titles, num_processes, timeout, sleep_freq, silent):
    pool = Pool(num_processes)
    results = []
    _ = [pool.apply_async(get_title_info,
                          args=(i, titles[i], timeout, i % sleep_freq == 0, silent),
                          callback=results.append)
         for i in range(len(titles))]
    pool.close()
    pool.join()

    return results


def clean_title(title):
    title = title. \
        replace(" @-@ ", "-"). \
        replace(" @.@ ", "."). \
        replace(" @,@ ", ","). \
        replace(" – ", "–")

    for char in ['.', ',', ':', ';', '!', '?', '\'', '"', '/', '\\', '...', ')']:
        title = title.replace(f" {char} ", f"{char} ")
        if title.startswith(f"{char} "):
            title = char + title[len(char)+1:]
        if title.endswith(f" {char}"):
            title = title[:-len(char)-1] + char

    title = title. \
        replace(" ( ", " ("). \
        replace(" '", "'")

    return title


def main(args):
    # load titles
    with open(args.title_map_file, "r", encoding='utf8') as fd:
        title_map = json.load(fd)

    # query wiki info
    clean_title_to_title = {
        clean_title(k): k
        for k in title_map.keys()
    }
    titles = [k for k in clean_title_to_title.keys()]
    results = query_titles(titles, args.num_processes, args.timeout,
                           args.sleep_freq, args.silent)

    # stats
    successful_query_titles = [result["query_title"] for result in results]
    missing_titles = [title for title in titles if title not in successful_query_titles]
    num_missing_titles_unk = len([title for title in missing_titles if "<unk>" in title])
    print(f"> successfully queried {len(results)} out of {len(titles)} titles.")
    print(f"> {num_missing_titles_unk} of missing titles are due to '<unk>' token.")
    print(f"> missing titles *not* because of an unknown token (first 50):\n",
          '\n'.join([title for title in missing_titles if "<unk>" not in title][:50]))

    # write results
    for result in results:
        del result["requests"]
        title_map[clean_title_to_title[result["query_title"]]].update(result)

    with open(args.output_path, "w", encoding='utf8') as fd:
        json.dump(title_map, fd, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('title_map_file', type=str, default='',
                        help='path to input file with titles separated by new lines')
    parser.add_argument('output_path', type=str, default='',
                        help='path to output file')
    parser.add_argument('--num_processes', type=int, default=10,
                        help='number of processes to parallelizing queries')
    parser.add_argument('--sleep_freq', type=int, default=1,
                        help='number of queries to sleep after')
    parser.add_argument('--timeout', type=int, default=5,
                        help='number of queries to sleep after')
    parser.add_argument('--silent', action='store_true', default=False)
    args = parser.parse_args()

    assert os.path.exists(args.title_map_file)

    main(args)

