import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest

from dacite import Config, MissingValueError, UnexpectedDataError, WrongTypeError, from_dict

typed_dict_support = pytest.mark.skipif(sys.version_info < (3, 8), reason="requires TypedDict")


@typed_dict_support
def test_from_dict_with_typed_dict_field():
    from typing import TypedDict

    class Point(TypedDict):
        x: int
        y: int

    @dataclass
    class Shape:
        origin: Point

    result = from_dict(Shape, {"origin": {"x": 1, "y": 2}})

    assert result == Shape(origin={"x": 1, "y": 2})
    assert type(result.origin) is dict


@typed_dict_support
def test_from_dict_with_total_false_and_explicit_none():
    from typing import NotRequired, TypedDict

    class Payload(TypedDict, total=False):
        value: Optional[str]
        name: NotRequired[str]

    @dataclass
    class Wrapper:
        payload: Payload

    result = from_dict(Wrapper, {"payload": {"value": None}})

    assert result == Wrapper(payload={"value": None})


@typed_dict_support
def test_from_dict_does_not_fill_missing_typed_dict_keys():
    from typing import TypedDict

    class Payload(TypedDict, total=False):
        value: Optional[str]

    @dataclass
    class Wrapper:
        payload: Payload

    result = from_dict(Wrapper, {"payload": {}})

    assert result.payload == {}


@typed_dict_support
def test_from_dict_with_inherited_typed_dict_required_redeclaration():
    from typing import Required, TypedDict

    class Base(TypedDict, total=False):
        optional: str

    class Child(Base):
        required: Required[int]

    @dataclass
    class Wrapper:
        child: Child

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"child": {"optional": "text"}})

    assert exception_info.value.field_path == "child.required"


@typed_dict_support
def test_from_dict_recursively_converts_typed_dict_nested_field():
    from typing import TypedDict

    class Inner(TypedDict):
        value: int

    class Outer(TypedDict):
        inner: Inner

    @dataclass
    class Wrapper:
        outer: Outer

    result = from_dict(Wrapper, {"outer": {"inner": {"value": "1"}}}, Config(cast=[int]))

    assert result == Wrapper(outer={"inner": {"value": 1}})


@typed_dict_support
def test_from_dict_passes_type_hooks_to_typed_dict_fields():
    from typing import TypedDict

    class Payload(TypedDict):
        value: str

    @dataclass
    class Wrapper:
        payload: Payload

    result = from_dict(Wrapper, {"payload": {"value": "TEXT"}}, Config(type_hooks={str: str.lower}))

    assert result == Wrapper(payload={"value": "text"})


@typed_dict_support
def test_from_dict_with_strict_typed_dict():
    from typing import TypedDict

    class Payload(TypedDict):
        value: str

    @dataclass
    class Wrapper:
        payload: Payload

    with pytest.raises(UnexpectedDataError) as exception_info:
        from_dict(Wrapper, {"payload": {"value": "text", "extra": 1}}, Config(strict=True))

    assert exception_info.value.field_path == "payload"
    assert exception_info.value.keys == {"extra"}


@typed_dict_support
def test_from_dict_with_typed_dict_forward_reference():
    from typing import TypedDict

    class Outer(TypedDict):
        inner: "Inner"

    class Inner(TypedDict):
        value: int

    @dataclass
    class Wrapper:
        outer: Outer

    result = from_dict(Wrapper, {"outer": {"inner": {"value": 1}}}, Config(forward_references={"Inner": Inner}))

    assert result == Wrapper(outer={"inner": {"value": 1}})


@typed_dict_support
def test_from_dict_with_typed_dict_in_optional():
    from typing import TypedDict

    class Payload(TypedDict):
        value: int

    @dataclass
    class Wrapper:
        payload: Optional[Payload]

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"payload": {}})

    assert exception_info.value.field_path == "payload.value"


@typed_dict_support
def test_from_dict_with_typed_dict_in_list():
    from typing import TypedDict

    class Payload(TypedDict):
        value: int

    @dataclass
    class Wrapper:
        payloads: List[Payload]

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"payloads": [{"value": 1}, {}]})

    assert exception_info.value.field_path == "payloads[1].value"


@typed_dict_support
def test_from_dict_with_typed_dict_in_dataclass():
    from typing import TypedDict

    class Payload(TypedDict):
        value: int

    @dataclass
    class Nested:
        payload: Payload

    @dataclass
    class Wrapper:
        nested: Nested

    with pytest.raises(WrongTypeError) as exception_info:
        from_dict(Wrapper, {"nested": {"payload": {"value": "text"}}})

    assert exception_info.value.field_path == "nested.payload.value"


@typed_dict_support
def test_from_dict_with_typed_dict_in_dict():
    from typing import TypedDict

    class Payload(TypedDict):
        value: int

    @dataclass
    class Wrapper:
        payloads: Dict[str, Payload]

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"payloads": {"first": {}}})

    assert exception_info.value.field_path == "payloads['first'].value"


@typed_dict_support
def test_from_dict_does_not_share_typed_dict_containers_between_inputs():
    from typing import TypedDict

    class Inner(TypedDict):
        values: List[int]

    class Outer(TypedDict):
        inner: Inner

    @dataclass
    class Wrapper:
        outer: Outer

    data = {"outer": {"inner": {"values": [1]}}}
    first = from_dict(Wrapper, data)
    second = from_dict(Wrapper, data)

    first.outer["inner"]["values"].append(2)

    assert second.outer["inner"]["values"] == [1]
    assert first.outer is not second.outer
    assert first.outer["inner"] is not second.outer["inner"]


@typed_dict_support
def test_from_dict_with_wrong_typed_dict_field_type():
    from typing import TypedDict

    class Payload(TypedDict):
        value: int

    @dataclass
    class Wrapper:
        payload: Payload

    with pytest.raises(WrongTypeError) as exception_info:
        from_dict(Wrapper, {"payload": {"value": "text"}})

    assert exception_info.value.field_path == "payload.value"
