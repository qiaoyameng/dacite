from dataclasses import dataclass
from typing import Annotated, List, Optional

import pytest

from dacite import (
    from_dict,
    Config,
    AnnotatedHandler,
    AnnotatedPhase,
    AnnotatedValidationError,
    AnnotatedMetadataError,
)


@dataclass(frozen=True)
class Strip:
    pass


@dataclass(frozen=True)
class MinLen:
    length: int


@dataclass(frozen=True)
class Raw:
    pass


@dataclass(frozen=True)
class Positive:
    pass


def make_config(log):
    def strip(value):
        log.append(("transform", "strip", value))
        return value.strip()

    def min_len_2(value):
        log.append(("validate", "min_len_2", value))
        if len(value) < 2:
            raise ValueError("too short")
        return value

    def min_len_5(value):
        log.append(("validate", "min_len_5", value))
        if len(value) < 5:
            raise ValueError("too short")
        return value

    def keep_raw(value):
        log.append(("validate", "keep_raw", value))
        return value

    def positive(value):
        log.append(("validate", "positive", value))
        if value <= 0:
            raise ValueError("not positive")
        return value

    return Config(
        annotated_handlers={
            Strip(): AnnotatedHandler.transform(strip),
            MinLen(2): AnnotatedHandler.validate(min_len_2),
            MinLen(5): AnnotatedHandler.validate(min_len_5),
            Raw(): AnnotatedHandler.validate(keep_raw),
            Positive(): AnnotatedHandler.validate(positive),
        }
    )


def test_annotated_transform_before_validate_and_nested_list():
    @dataclass
    class X:
        name: Annotated[str, Strip(), MinLen(2)]
        note: Annotated[str, Raw()]
        tags: List[Annotated[str, Strip()]]

    log = []
    result = from_dict(
        X,
        {"name": "  hello  ", "note": "  keep me  ", "tags": [" a ", " b"]},
        make_config(log),
    )

    assert result.name == "hello"
    assert result.note == "  keep me  "
    assert result.tags == ["a", "b"]
    assert log == [
        ("transform", "strip", "  hello  "),
        ("validate", "min_len_2", "hello"),
        ("validate", "keep_raw", "  keep me  "),
        ("transform", "strip", " a "),
        ("transform", "strip", " b"),
    ]


def test_annotated_validation_error_keeps_field_path_and_phase():
    @dataclass
    class X:
        name: Annotated[str, Strip(), MinLen(2)]

    with pytest.raises(AnnotatedValidationError) as exc_info:
        from_dict(X, {"name": "  x  "}, make_config([]))

    assert exc_info.value.field_path == "name"
    assert exc_info.value.phase == AnnotatedPhase.VALIDATE
    assert exc_info.value.metadata == MinLen(2)
    assert exc_info.value.value == "x"


def test_annotated_failure_leaves_no_state_for_next_conversion():
    @dataclass
    class X:
        name: Annotated[str, Strip(), MinLen(2)]
        tags: List[Annotated[str, Strip()]]

    log = []
    config = make_config(log)

    with pytest.raises(AnnotatedValidationError):
        from_dict(X, {"name": " x ", "tags": [" a "]}, config)

    log.clear()
    result = from_dict(X, {"name": " ok ", "tags": [" a "]}, config)

    assert result == X(name="ok", tags=["a"])
    assert log == [
        ("transform", "strip", " ok "),
        ("validate", "min_len_2", "ok"),
        ("transform", "strip", " a "),
    ]


def test_annotated_same_type_different_metadata_not_shared():
    @dataclass
    class X:
        short: Annotated[str, MinLen(2)]
        long: Annotated[str, MinLen(5)]

    log = []
    config = make_config(log)

    result = from_dict(X, {"short": "ab", "long": "abcde"}, config)
    assert result == X(short="ab", long="abcde")

    with pytest.raises(AnnotatedValidationError) as exc_info:
        from_dict(X, {"short": "abc", "long": "abc"}, config)

    assert exc_info.value.field_path == "long"
    assert exc_info.value.metadata == MinLen(5)


def test_annotated_unknown_metadata_kept_by_default():
    @dataclass
    class X:
        s: Annotated[str, "unknown-marker"]

    result = from_dict(X, {"s": "value"})

    assert result == X(s="value")


def test_annotated_unknown_metadata_rejected_in_strict_mode():
    @dataclass
    class X:
        s: Annotated[str, "unknown-marker"]

    with pytest.raises(AnnotatedMetadataError) as exc_info:
        from_dict(X, {"s": "value"}, Config(strict_annotated=True))

    assert exc_info.value.field_path == "s"
    assert exc_info.value.metadata == "unknown-marker"


def test_annotated_with_optional_and_type_hook_runs_each_handler_once():
    @dataclass
    class X:
        count: Optional[Annotated[int, Positive()]]
        name: Annotated[str, Strip()]

    log = []

    def int_hook(value):
        log.append(("hook", "int", value))
        return int(value)

    config = make_config(log)
    config.type_hooks[int] = int_hook

    result = from_dict(X, {"count": "5", "name": " x "}, config)

    assert result == X(count=5, name="x")
    assert log == [
        ("hook", "int", "5"),
        ("validate", "positive", 5),
        ("transform", "strip", " x "),
    ]


def test_annotated_optional_none_skips_handlers():
    @dataclass
    class X:
        count: Optional[Annotated[int, Positive()]]

    log = []
    result = from_dict(X, {"count": None}, make_config(log))

    assert result == X(count=None)
    assert log == []
