import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='./wiki_dataset', type=str, help='The output path to store data')
    args = parser.parse_args()

    wikitext_dataset = load_dataset('wikitext', 'wikitext-2-v1')
    wikitext_dataset.save_to_disk(args.output)

if __name__ == '__main__':
    main()