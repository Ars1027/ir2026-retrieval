from pipeline.hybrid import solve_method


def solve(question, tools, config):
    return solve_method(question, tools, config, "hybrid")
