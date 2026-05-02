"""
2_extract_comments_only.py
===========================
Reads any Instagram comments CSV and outputs a new CSV that contains
ONLY the comment_text column — dropping everything else.

Input  : comments CSV with at least a 'comment_text' column
Output : comments_only.csv  (one column: comment_text)

Usage
-----
    python 2_extract_comments_only.py
    python 2_extract_comments_only.py --input my_comments.csv --output stripped.csv
"""

import argparse
import sys
import pandas as pd


def extract_comments_only(
    input_path: str = "comments.csv",
    output_path: str = "comments_only.csv",
) -> pd.DataFrame:
    """
    Load a comments CSV and keep only the comment_text column.

    Parameters
    ----------
    input_path  : str — path to the source CSV file
    output_path : str — where to save the stripped CSV

    Returns
    -------
    pd.DataFrame with a single 'comment_text' column
    """
    df = pd.read_csv(input_path, encoding="utf-8")

    if "comment_text" not in df.columns:
        raise ValueError(
            f"'comment_text' column not found in {input_path}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    comments_df = df[["comment_text"]].copy()

    # Report what was dropped
    dropped = [c for c in df.columns if c != "comment_text"]
    print(f"Loaded   : {len(df)} rows from '{input_path}'")
    print(f"Dropped  : {dropped}")
    print(f"Kept     : ['comment_text']")

    comments_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved    : '{output_path}'  ({len(comments_df)} rows)\n")

    return comments_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strip a comments CSV down to the comment_text column only."
    )
    parser.add_argument(
        "--input",  "-i",
        default="comments_20260501_230423.csv",
        help="Path to the input CSV  (default: comments_20260501_230423.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="comments_only.csv",
        help="Path for the output CSV (default: comments_only.csv)",
    )
    args = parser.parse_args()

    try:
        df = extract_comments_only(args.input, args.output)
        print(df.head(10).to_string(index=False))
    except FileNotFoundError:
        print(f"Error: file not found — '{args.input}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
