"""Name cleanup and prompt text."""
import re

from .config import AMBIGUOUS, QUALIFIER


def legalize(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or AMBIGUOUS


def pretty(slug):
    return slug.replace("_", " ").title()


def base_name(tag):
    return re.sub(r"_?\([^()]*\)", "", tag)


def title_name(tag):
    return base_name(tag.split("/")[0])


def describe(tag):
    qualifiers = QUALIFIER.findall(tag)
    name = pretty(legalize(base_name(tag)))
    return f"an illustration of {name} from {pretty(legalize(title_name(qualifiers[-1])))}" if qualifiers else f"an illustration of {name}"


def describe_series(tag):
    return f"an illustration of a character from {pretty(legalize(base_name(tag)))}"
