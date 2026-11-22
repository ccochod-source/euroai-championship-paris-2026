"""Call the official client with the team token on stdin and a longer upload timeout."""

import argparse
import sys

import requests
import submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("Team token must be supplied on standard input")

    post = requests.post

    def upload(url, *, timeout, **kwargs):
        return post(url, timeout=(120, timeout[1]), allow_redirects=False, **kwargs)

    submission.requests.post = upload
    submission.SUBMISSION_FILE = args.checkpoint
    submission.API_KEY = token
    return submission.main()


if __name__ == "__main__":
    raise SystemExit(main())
