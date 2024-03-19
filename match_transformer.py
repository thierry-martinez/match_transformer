import dataclasses
import functools
import logging
import sys
import typing

import libcst as cst


def node_compare(
    left: cst.BaseExpression, operator: cst.BaseCompOp, right: cst.BaseExpression
) -> cst.BooleanOperation:
    return cst.Comparison(
        left=left,
        comparisons=[cst.ComparisonTarget(operator=operator, comparator=right)],
    )


def node_equal(
    left: cst.BaseExpression, right: cst.BaseExpression
) -> cst.BooleanOperation:
    return node_compare(left, cst.Equal(), right)


def node_is(
    left: cst.BaseExpression, right: cst.BaseExpression
) -> cst.BooleanOperation:
    return node_compare(left, cst.Is(), right)


def node_not(value: cst.BaseExpression) -> cst.BaseExpression:
    return cst.UnaryOperation(operator=cst.Not(), expression=value)


def node_and(a: cst.BaseExpression, b: cst.BaseExpression) -> cst.BooleanOperation:
    return cst.BooleanOperation(left=a, operator=cst.And(), right=b)


def node_or(a: cst.BaseExpression, b: cst.BaseExpression) -> cst.BooleanOperation:
    return cst.BooleanOperation(left=a, operator=cst.Or(), right=b)


def node_index(
    col: cst.BaseExpression, index: cst.BaseExpression
) -> cst.BooleanOperation:
    return cst.Subscript(value=col, slice=[cst.SubscriptElement(cst.Index(index))])


def node_slice(
    col: cst.BaseExpression, lower: cst.BaseExpression, upper: cst.BaseExpression
) -> cst.BooleanOperation:
    return cst.Subscript(
        value=col, slice=[cst.SubscriptElement(cst.Slice(lower, upper))]
    )


def node_list(elements: list[cst.BaseExpression]) -> cst.BaseExpression:
    return cst.List(elements=[cst.Element(value) for value in elements])


def node_set(elements: list[cst.BaseExpression]) -> cst.BaseExpression:
    if elements:
        return cst.Set(elements=[cst.Element(value) for value in elements])
    else:
        return cst.Call(cst.Name("set"), [])


def node_isinstance(
    value: cst.BaseExpression, c: cst.BaseExpression
) -> cst.BaseExpression:
    return cst.Call(cst.Name("isinstance"), [cst.Arg(value), cst.Arg(c)])


def node_int(value: int) -> cst.Integer:
    if value < 0:
        return cst.UnaryOperation(
            operator=cst.Minus(), expression=cst.Integer(str(-value))
        )
    else:
        return cst.Integer(str(value))


def node_len(value: cst.BaseExpression) -> cst.Call:
    return cst.Call(cst.Name("len"), args=[cst.Arg(value)])


def node_getattr(
    value: cst.BaseExpression,
    attr: cst.BaseExpression,
    default: cst.BaseExpression = None,
) -> cst.Call:
    args = [cst.Arg(value), cst.Arg(attr)]
    if default:
        args.append(cst.Arg(default))
    return cst.Call(cst.Name("getattr"), args=args)


def node_par(node: cst.BaseExpression) -> cst.BaseExpression:
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def node_qualified_name(*components: list[str]) -> cst.BaseExpression:
    return functools.reduce(cst.Attribute, map(cst.Name, components))


def node_str(s: str) -> cst.SimpleString:
    return cst.SimpleString(repr(s))


class NonLinearPattern(Exception):
    def __init__(self, pattern: cst.MatchPattern):
        self.__pattern = pattern

    def __str__(self) -> str:
        return f"Non-linear pattern: {self.__pattern}"


def add_bindings(
    target: dict[str, cst.BaseExpression], source: dict[str, cst.BaseExpression]
) -> None:
    for name, value in source.items():
        if name in target:
            raise NonLinearPattern(pattern)
        target[name] = value


def check_sequence_or_mapping(
    subject: cst.BaseExpression, sequence: bool
) -> cst.BaseExpression:
    return node_and(
        node_isinstance(
            subject,
            node_qualified_name(
                "collections", "abc", "Sequence" if sequence else "Mapping"
            ),
        ),
        node_compare(
            cst.Call(
                cst.Name("next"),
                [
                    cst.Arg(
                        cst.GeneratorExp(
                            cst.Name("parent"),
                            for_in=cst.CompFor(
                                cst.Name("parent"),
                                cst.Attribute(
                                    cst.Attribute(subject, cst.Name("__class__")),
                                    cst.Name("__mro__"),
                                ),
                                [
                                    cst.CompIf(
                                        node_or(
                                            node_compare(
                                                cst.Name("parent"),
                                                cst.Is(),
                                                cst.Name("dict"),
                                            ),
                                            node_or(
                                                node_compare(
                                                    cst.Name("parent"),
                                                    cst.Is(),
                                                    cst.Name("list"),
                                                ),
                                                node_or(
                                                    node_compare(
                                                        cst.Name("parent"),
                                                        cst.Is(),
                                                        node_qualified_name(
                                                            "collections",
                                                            "abc",
                                                            "Mapping",
                                                        ),
                                                    ),
                                                    node_compare(
                                                        cst.Name("parent"),
                                                        cst.Is(),
                                                        node_qualified_name(
                                                            "collections",
                                                            "abc",
                                                            "Sequence",
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        )
                                    )
                                ],
                            ),
                        )
                    ),
                    cst.Arg(cst.Name("None")),
                ],
            ),
            cst.NotIn(),
            (
                node_list(
                    [
                        cst.Name("dict"),
                        node_qualified_name("collections", "abc", "Mapping"),
                    ]
                )
                if sequence
                else node_list(
                    [
                        cst.Name("list"),
                        node_qualified_name("collections", "abc", "Sequence"),
                    ]
                )
            ),
        ),
    )


@dataclasses.dataclass
class MatchTest:
    prechecks: list(tuple[cst.BaseExpression, cst.CSTNode])
    test: cst.CSTNode | None
    bindings: dict[str, cst.BaseExpression]

    @staticmethod
    def from_pattern(subject: cst.CSTNode, pattern: cst.CSTNode | None) -> "MatchTest":
        match subject, pattern:
            case _, None | cst.MatchAs(pattern=None, name=None):
                return MatchTest(prechecks=[], test=None, bindings=dict())
            case _, cst.MatchAs(pattern=pattern, name=name):
                assert name is not None
                result = MatchTest.from_pattern(subject, pattern)
                result.bindings[name.value] = subject
                return result
            case cst.Tuple(values), cst.MatchList(patterns) | cst.MatchTuple(patterns):
                star = next(
                    (
                        (i, pattern)
                        for i, pattern in enumerate(patterns)
                        if isinstance(pattern, cst.MatchStar)
                    ),
                    None,
                )
                bindings = dict()
                if star is not None:
                    i, pattern = star
                    rhs_count = len(patterns) - i - 1
                    if pattern.name is not None:
                        bindings[pattern.name.value] = cst.List(
                            values[i : i + len(values) - len(patterns) + 1]
                        )
                    values = values[0:i] + values[len(values) - rhs_count :]
                    patterns = patterns[0:i] + patterns[i + 1 :]
                match_tests = [
                    MatchTest.from_pattern(subject.value, pattern.value)
                    for subject, pattern in zip(values, patterns)
                ]
                tests = [
                    match_test.test
                    for match_test in match_tests
                    if match_test.test is not None
                ]
                if tests:
                    test = functools.reduce(
                        node_and,
                        tests,
                    )
                else:
                    test = None
                for match_test in match_tests:
                    add_bindings(bindings, match_test.bindings)
                return MatchTest(prechecks=[], test=test, bindings=bindings)
            case _, cst.MatchList(patterns) | cst.MatchTuple(patterns):
                star = next(
                    (
                        (i, pattern)
                        for i, pattern in enumerate(patterns)
                        if isinstance(pattern, cst.MatchStar)
                    ),
                    None,
                )
                bindings = dict()
                test = node_and(
                    check_sequence_or_mapping(subject, True),
                    node_not(
                        node_isinstance(
                            subject,
                            cst.Tuple(
                                [
                                    cst.Element(cst.Name(cls))
                                    for cls in ("str", "bytes", "bytearray")
                                ]
                            ),
                        )
                    ),
                )
                subject_len = node_len(subject)
                if star is None:
                    test = node_and(
                        test, node_equal(subject_len, node_int(len(patterns)))
                    )
                else:
                    i, pattern = star
                    rhs_count = len(patterns) - i - 1
                    if pattern.name is not None:
                        if rhs_count > 0:
                            upper = node_int(-rhs_count)
                        else:
                            upper = None
                        bindings[pattern.name.value] = cst.Call(
                            cst.Name("list"),
                            [cst.Arg(node_slice(subject, node_int(i), upper))],
                        )
                    patterns = patterns[0:i] + patterns[i + 1 :]
                    min_len = i + rhs_count
                    if min_len > 0:
                        test = node_and(
                            test,
                            node_compare(
                                subject_len, cst.GreaterThanEqual(), node_int(min_len)
                            ),
                        )
                for i, pattern in enumerate(patterns):
                    if star is None:
                        index = node_int(i)
                    else:
                        j, _ = star
                        if i >= j:
                            index = cst.BinaryOperation(
                                subject_len, cst.Subtract(), node_int(len(patterns) - i)
                            )
                        else:
                            index = node_int(i)
                    match_test = MatchTest.from_pattern(
                        node_index(subject, index), pattern.value
                    )
                    if match_test.test is not None:
                        test = node_and(test, match_test.test)
                    add_bindings(bindings, match_test.bindings)
                return MatchTest(prechecks=[], test=test, bindings=bindings)
            case _, cst.MatchMapping(elements=elements, rest=rest):
                keys = [element.key for element in elements]
                bindings = dict()
                # test = node_isinstance(subject, node_qualified_name("collections", "abc", "Mapping"))
                test = check_sequence_or_mapping(subject, False)
                subject_keys = cst.Call(cst.Attribute(subject, cst.Name("keys")), [])
                if rest:
                    bindings[rest.value] = cst.DictComp(
                        key=cst.Name("key"),
                        value=cst.Name("value"),
                        for_in=cst.CompFor(
                            target=cst.Tuple(
                                [
                                    cst.Element(cst.Name("key")),
                                    cst.Element(cst.Name("value")),
                                ]
                            ),
                            iter=cst.Call(
                                cst.Attribute(subject, cst.Name("items")), []
                            ),
                            ifs=[
                                cst.CompIf(
                                    cst.Comparison(
                                        cst.Name("key"),
                                        [
                                            cst.ComparisonTarget(
                                                cst.NotIn(), node_set(keys)
                                            )
                                        ],
                                    )
                                )
                            ],
                        ),
                    )
                prechecks = [
                    (
                        node_and(
                            test,
                            node_compare(
                                node_len(node_set(keys)),
                                cst.LessThan(),
                                node_len(node_list(keys)),
                            ),
                        ),
                        cst.IndentedBlock(
                            [
                                cst.SimpleStatementLine(
                                    [
                                        cst.Raise(
                                            cst.Call(
                                                cst.Name("ValueError"),
                                                [cst.Arg(node_str("Duplicate keys"))],
                                            )
                                        )
                                    ],
                                    trailing_whitespace=cst.TrailingWhitespace(
                                        newline=cst.Newline()
                                    ),
                                )
                            ]
                        ),
                    )
                ]
                for element in elements:
                    test = node_and(test, node_compare(element.key, cst.In(), subject))
                    match_test = MatchTest.from_pattern(
                        node_index(subject, element.key), element.pattern
                    )
                    if match_test.test:
                        test = node_and(test, match_test.test)
                    add_bindings(bindings, match_test.bindings)
                return MatchTest(prechecks, test=test, bindings=bindings)
            case _, cst.MatchValue(value):
                test = node_equal(subject, value)
                return MatchTest(prechecks=[], test=test, bindings=dict())
            case _, cst.MatchSingleton(value):
                test = node_is(subject, value)
                return MatchTest(prechecks=[], test=test, bindings=dict())
            case _, cst.MatchOr(patterns):
                match_tests = [
                    MatchTest.from_pattern(subject, pattern.pattern)
                    for pattern in patterns
                ]
                if any(match_test.test == None for match_test in match_tests):
                    test = None
                else:
                    test = functools.reduce(
                        node_or, [match_test.test for match_test in match_tests]
                    )
                bindings = dict()
                for match_test in reversed(match_tests):
                    for name, value in match_test.bindings.items():
                        if match_test.test:
                            orelse = bindings.get(name, cst.Name("None"))
                            value = cst.IfExp(
                                test=match_test.test, body=value, orelse=orelse
                            )
                        bindings[name] = value
                return MatchTest(prechecks=[], test=test, bindings=bindings)
            case _, cst.MatchClass(cls, patterns=patterns, kwds=kwds):
                test = node_isinstance(subject, cls)
                match_args = node_str("__match_args__")
                match patterns, cls.value:
                    case (
                        [pattern],
                        "bool"
                        | "bytearray"
                        | "bytes"
                        | "dict"
                        | "float"
                        | "frozenset"
                        | "int"
                        | "list"
                        | "set"
                        | "str"
                        | "tuple",
                    ):
                        prechecks = []
                        match_test = MatchTest.from_pattern(subject, patterns[0].value)
                        if match_test.test:
                            test = node_and(test, match_test.test)
                        bindings = match_test.bindings
                    case _:
                        bindings = dict()
                        static_key_list = cst.Tuple(
                            [
                                cst.Element(node_str(keyword.key.value))
                                for keyword in kwds
                            ]
                        )
                        if patterns:
                            match_args = node_getattr(cls, match_args, cst.Tuple([]))
                            key_list = cst.BinaryOperation(
                                node_slice(
                                    match_args, node_int(0), node_int(len(patterns))
                                ),
                                cst.Add(),
                                static_key_list,
                            )
                            prechecks = [
                                (
                                    node_and(
                                        test,
                                        node_compare(
                                            node_len(match_args),
                                            cst.LessThan(),
                                            node_int(len(patterns)),
                                        ),
                                    ),
                                    cst.IndentedBlock(
                                        [
                                            cst.SimpleStatementLine(
                                                [
                                                    cst.Raise(
                                                        cst.Call(
                                                            cst.Name("TypeError"),
                                                            [
                                                                cst.Arg(
                                                                    cst.FormattedString(
                                                                        [
                                                                            cst.FormattedStringText(
                                                                                f"{cls.value}() accepts "
                                                                            ),
                                                                            cst.FormattedStringExpression(
                                                                                node_len(
                                                                                    match_args
                                                                                )
                                                                            ),
                                                                            cst.FormattedStringText(
                                                                                f" positional sub-patterns ({len(patterns)} given)"
                                                                            ),
                                                                        ]
                                                                    )
                                                                )
                                                            ],
                                                        )
                                                    )
                                                ],
                                                trailing_whitespace=cst.TrailingWhitespace(
                                                    newline=cst.Newline()
                                                ),
                                            )
                                        ]
                                    ),
                                )
                            ]
                            for i, pattern in enumerate(patterns):
                                match_test = MatchTest.from_pattern(
                                    node_getattr(
                                        subject, node_index(match_args, node_int(i))
                                    ),
                                    pattern.value,
                                )
                                if match_test.test is not None:
                                    test = node_and(test, match_test.test)
                                add_bindings(bindings, match_test.bindings)
                        else:
                            prechecks = []
                            key_list = static_key_list
                        prechecks.append(
                            (
                                node_and(
                                    test,
                                    node_compare(
                                        node_len(
                                            cst.Call(
                                                cst.Name("set"), [cst.Arg(key_list)]
                                            )
                                        ),
                                        cst.LessThan(),
                                        node_len(key_list),
                                    ),
                                ),
                                cst.IndentedBlock(
                                    [
                                        cst.SimpleStatementLine(
                                            [
                                                cst.Raise(
                                                    cst.Call(
                                                        cst.Name("TypeError"),
                                                        [
                                                            cst.Arg(
                                                                node_str(
                                                                    "Duplicate keys"
                                                                )
                                                            )
                                                        ],
                                                    )
                                                )
                                            ],
                                            trailing_whitespace=cst.TrailingWhitespace(
                                                newline=cst.Newline()
                                            ),
                                        )
                                    ]
                                ),
                            )
                        )
                        for keyword in kwds:
                            match_test = MatchTest.from_pattern(
                                node_getattr(subject, node_str(keyword.key.value)),
                                keyword.pattern,
                            )
                            if match_test.test is not None:
                                test = node_and(test, match_test.test)
                            add_bindings(bindings, match_test.bindings)
                return MatchTest(prechecks, test=test, bindings=bindings)
            case _:
                raise Exception(f"Unsupported: match {subject} against {pattern}")


@dataclasses.dataclass
class Case:
    match_test: MatchTest
    guard: cst.BaseExpression | None
    body: cst.CSTNode


def del_var(name: str) -> cst.CSTNode:
    return cst.SimpleStatementLine(body=[cst.Del(cst.Name(name))])


def pure(node: cst.CSTNode) -> bool:
    match node:
        case cst.BaseNumber() | cst.SimpleString():
            return True
        case cst.Tuple(elements):
            return all(pure(element.value) for element in elements)
        case _:
            return False


def append_block(target: list[cst.CSTNode], node: cst.CSTNode) -> None:
    match node:
        case cst.IndentedBlock(body):
            target.extend(body)
        case _:
            target.append(node)


def node_orelse(body: cst.CSTNode) -> cst.CSTNode:
    match body:
        case cst.If():
            return body
        case _:
            return cst.Else(body)


class MatchTransformer(cst.CSTTransformer):
    def __init__(self, module):
        self.__module = module

    def replace_match(self, node: cst.Match) -> list[cst.CSTNode]:
        del_subject = del_var("subject")
        result = del_subject
        subject_var = cst.Name("subject")
        prefix = [
            cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(subject_var)],
                        value=node_par(node.subject),
                    )
                ]
            )
        ]
        subject = subject_var
        cases = [
            Case(
                match_test=MatchTest.from_pattern(subject, c.pattern),
                guard=c.guard,
                body=c.body,
            )
            for c in node.cases
        ]
        for i, c in reversed(list(enumerate(cases))):
            if c.match_test.bindings:
                binding_statement = cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(target=cst.Name(name))],
                            value=value,
                        )
                        for name, value in c.match_test.bindings.items()
                    ]
                )
            else:
                binding_statement = None
            if c.match_test.test and binding_statement and c.guard:
                result_block = []
                result_block.append(
                    cst.SimpleStatementLine(
                        body=[
                            cst.Assign(
                                targets=[cst.AssignTarget(target=cst.Name("test"))],
                                value=c.match_test.test,
                            )
                        ]
                    )
                )
                result_block.append(cst.If(cst.Name("test"), body=binding_statement))
                del_test = del_var("test")
                body_block = [del_subject, del_test]
                append_block(body_block, c.body)
                orelse_block = [del_test]
                append_block(orelse_block, result)
                result_block.append(
                    cst.If(
                        node_and(cst.Name("test"), c.guard),
                        body=cst.IndentedBlock(body_block),
                        orelse=cst.Else(cst.IndentedBlock(orelse_block)),
                    )
                )
                result = cst.IndentedBlock(result_block)
            else:
                body_block = [del_subject]
                append_block(body_block, c.body)
                if binding_statement:
                    block = [binding_statement]
                    if c.guard:
                        block.append(
                            cst.If(
                                c.guard,
                                body=cst.IndentedBlock(body_block),
                                orelse=node_orelse(result),
                            )
                        )
                    else:
                        block.extend(body_block)
                    node = cst.IndentedBlock(block)
                    if c.match_test.test:
                        result = cst.If(c.match_test.test, node, node_orelse(result))
                    else:
                        result = node
                else:
                    if c.match_test.test and c.guard:
                        test = node_and(c.match_test.test, c.guard)
                    elif c.match_test.test:
                        test = c.match_test.test
                    elif c.guard:
                        test = c.guard
                    else:
                        test = None
                    body = cst.IndentedBlock(body_block)
                    if test:
                        result = cst.If(test, body=body, orelse=node_orelse(result))
                    else:
                        result = body
            for precheck, body in reversed(c.match_test.prechecks):
                result = cst.If(precheck, body, orelse=node_orelse(result))
        match result:
            case cst.IndentedBlock(body=body):
                return prefix + list(body)
            case _:
                return prefix + [result]

    def replace_match_in_sequence(
        self, sequence: list[cst.CSTNode]
    ) -> list[cst.CSTNode]:
        result = None
        for index, node in enumerate(sequence):
            match node:
                case cst.Match(_):
                    if result is None:
                        result = list(sequence[0:index])
                    result.extend(self.replace_match(node))
                case _:
                    if result is not None:
                        result.append(node)
        return result

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.CSTNode:
        body = self.replace_match_in_sequence(updated_node.body)
        if body is None:
            return updated_node
        return updated_node.with_changes(body=body)

    def leave_IndentedBlock(
        self, original_node: cst.IndentedBlock, updated_node: cst.IndentedBlock
    ) -> cst.CSTNode:
        body = self.replace_match_in_sequence(updated_node.body)
        if body is None:
            return updated_node
        return updated_node.with_changes(body=body)


def main() -> None:
    m = cst.parse_module(sys.stdin.read())
    transformer = MatchTransformer(m)
    print(m.visit(transformer).code)


if __name__ == "__main__":
    main()
