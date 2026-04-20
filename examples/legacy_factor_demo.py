from __future__ import annotations

import argparse

from QUANTAXIS.QAFactor import QAFeatureView


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal demo for the legacy QUANTAXIS factor stack."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-factors", help="List factor tables in ClickHouse.")

    show = subparsers.add_parser("show-factor", help="Show rows from one factor table.")
    show.add_argument("factor", help="Factor table name in the factor database.")
    show.add_argument("--start", help="Start date like 2020-01-01.")
    show.add_argument("--end", help="End date like 2020-12-31.")
    show.add_argument("--head", type=int, default=10, help="Rows to print.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    view = QAFeatureView()

    if args.command == "list-factors":
        for name in view.get_all_factorname():
            print(name)
        return

    if args.command == "show-factor":
        frame = view.get_single_factor(args.factor, start=args.start, end=args.end)
        print(frame.head(args.head))
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
