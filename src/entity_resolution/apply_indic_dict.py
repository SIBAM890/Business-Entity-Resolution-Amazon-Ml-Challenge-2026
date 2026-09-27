"""Apply learned Indic dictionary to normalized data."""
import sys
from .indic_dictionary import load, apply


def main():
    mapping = load()
    for split in sys.argv[1:] or ["train", "test"]:
        apply(split, mapping)


if __name__ == "__main__":
    main()