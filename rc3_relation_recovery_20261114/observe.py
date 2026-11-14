"""Read only immutable publications; never read an active accounting file."""
import argparse
import json
from .persistence import observe_published


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    print(json.dumps(observe_published(args.stage), ensure_ascii=False))


if __name__ == '__main__':
    main()
