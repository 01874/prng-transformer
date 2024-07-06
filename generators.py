from collections.abc import Generator

# Using Knuth's params as default, from Numerical Recipes (per wikipedia https://en.wikipedia.org/wiki/Linear_congruential_generator)
def lcg(m: int=4294967296, a: int=1664525, c: int=1013904223, s: int=123456) -> Generator[int, None, None]:
    while True:
        s = (a * s + c) % m
        yield s