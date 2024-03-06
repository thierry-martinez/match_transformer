import functools
import sys

import libcst as cst


def convert_pattern_to_test(subject, pattern):
    match subject, pattern:
        case cst.Tuple(values), cst.MatchList(patterns):
            tests = [
                convert_pattern_to_test(subject.value, pattern.value)
                for subject, pattern in zip(values, patterns)
            ]
            return functools.reduce(
                lambda a, b: cst.BooleanOperation(left=a, operator=cst.And(), right=b),
                tests,
            )
        case _, cst.MatchValue(value):
            return cst.Comparison(
                left=subject,
                comparisons=[
                    cst.ComparisonTarget(operator=cst.Equal(), comparator=value)
                ],
            )
        case _:
            raise Exception(f"Unsupported: match {subject} against {pattern}")


class MatchTransformer(cst.CSTTransformer):
    def __init__(self, module):
        self.__module = module

    def leave_Match(
        self, original_node: cst.Match, updated_node: cst.Match
    ) -> cst.CSTNode:
        result = None
        for c in reversed(original_node.cases):
            test = convert_pattern_to_test(original_node.subject, c.pattern)
            result = cst.If(test=test, body=c.body, orelse=result)
        original_code = self.__module.code_for_node(original_node)
        comment_lines = [
            cst.EmptyLine(comment=cst.Comment(value=f"# {line}"))
            for line in original_code.splitlines()
        ]
        result = cst.If(
            test=result.test,
            body=result.body,
            orelse=result.orelse,
            leading_lines=comment_lines,
        )
        return result


m = cst.parse_module(sys.stdin.read())
transformer = MatchTransformer(m)
print(m.visit(transformer).code)
