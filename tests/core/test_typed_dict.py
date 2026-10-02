import sys
from dataclasses import dataclass
from typing import List, Optional

import pytest

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

try:
    from typing import NotRequired, Required
except ImportError:
    NotRequired = None
    Required = None

from dacite import Config, MissingValueError, UnexpectedDataError, WrongTypeError, from_dict

required_support = pytest.mark.skipif(sys.version_info < (3, 11), reason="requires Python 3.11")


class InnerTypedDict(TypedDict):
    value: int
    name: NotRequired[str] if NotRequired else str


class NestedTypedDict(TypedDict):
    inner: InnerTypedDict
    tags: NotRequired[List[str]] if NotRequired else List[str]


@dataclass
class NestedWrapper:
    payload: NestedTypedDict


@dataclass
class OptionalWrapper:
    payload: Optional[InnerTypedDict]


@dataclass
class ListWrapper:
    payloads: List[InnerTypedDict]


def test_from_dict_recursively_builds_typed_dict_as_plain_dict():
    @dataclass
    class Wrapper:
        payload: NestedTypedDict

    result = from_dict(Wrapper, {"payload": {"inner": {"value": 1, "name": "NAME"}, "tags": ["A", "B"]}})

    assert isinstance(result.payload, dict)
    assert type(result.payload) is dict
    assert type(result.payload["inner"]) is dict
    assert result == Wrapper(payload={"inner": {"value": 1, "name": "NAME"}, "tags": ["A", "B"]})


def test_from_dict_preserves_missing_optional_key_without_default():
    @dataclass
    class Wrapper:
        payload: InnerTypedDict

    result = from_dict(Wrapper, {"payload": {"value": 1}})

    assert result.payload == {"value": 1}
    assert "name" not in result.payload


def test_from_dict_ignores_extra_typed_dict_keys_without_strict_mode():
    result = from_dict(NestedWrapper, {"payload": {"inner": {"value": 1}, "extra": 2}})

    assert result.payload == {"inner": {"value": 1}}


def test_from_dict_preserves_none_for_optional_typed_dict_value():
    class OptionalValue(TypedDict, total=False):
        value: Optional[int]

    @dataclass
    class Wrapper:
        payload: OptionalValue

    missing_result = from_dict(Wrapper, {"payload": {}})
    none_result = from_dict(Wrapper, {"payload": {"value": None}})

    assert missing_result.payload == {}
    assert none_result.payload == {"value": None}


def test_from_dict_distinguishes_missing_required_optional_value_from_none():
    class RequiredOptionalValue(TypedDict):
        value: Optional[int]

    @dataclass
    class Wrapper:
        payload: RequiredOptionalValue

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"payload": {}})

    assert exception_info.value.field_path == "payload.value"
    assert from_dict(Wrapper, {"payload": {"value": None}}).payload == {"value": None}


def test_from_dict_validates_total_false_typed_dict():
    @dataclass
    class Wrapper:
        payload: InnerTypedDict

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"payload": {"name": "test"}})

    assert exception_info.value.field_path == "payload.value"


def test_from_dict_uses_redeclared_optional_requiredness_after_inheritance():
    class Base(TypedDict):
        required: int
        optional: NotRequired[str] if NotRequired else str

    class Redeclared(Base, total=False):
        required: int
        optional: str

    @dataclass
    class Wrapper:
        payload: Redeclared

    result = from_dict(Wrapper, {"payload": {}})

    assert result.payload == {}
    assert result.payload is not None
    assert "required" not in result.payload
    assert "optional" not in result.payload


@required_support
def test_from_dict_uses_redeclared_requiredness_after_inheritance():
    class Base(TypedDict, total=False):
        value: int

    class Redeclared(Base):
        value: Required[int]

    @dataclass
    class Wrapper:
        payload: Redeclared

    with pytest.raises(MissingValueError) as exception_info:
        from_dict(Wrapper, {"payload": {}})

    assert exception_info.value.field_path == "payload.value"
    assert from_dict(Wrapper, {"payload": {"value": 1}}).payload == {"value": 1}


def test_from_dict_recursively_builds_nested_dataclass_inside_typed_dict():
    @dataclass
    class Inner:
        value: int

    class Parent(TypedDict):
        inner: Inner

    @dataclass
    class Wrapper:
        payload: Parent

    result = from_dict(Wrapper, {"payload": {"inner": {"value": 1}}})

    assert result == Wrapper(payload={"inner": Inner(value=1)})


def test_from_dict_passes_type_hooks_to_nested_typed_dict_fields():
    result = from_dict(
        NestedWrapper,
        {"payload": {"inner": {"value": 1, "name": "name"}, "tags": ["tag"]}},
        Config(type_hooks={str: str.upper}),
    )

    assert result.payload["inner"]["name"] == "NAME"
    assert result.payload["tags"] == ["TAG"]


def test_from_dict_allows_type_hook_for_typed_dict_itself():
    result = from_dict(
        NestedWrapper,
        {"payload": {"inner": {"value": 1}}},
        Config(type_hooks={NestedTypedDict: lambda payload: {"inner": payload["inner"], "tags": ["hooked"]}}),
    )

    assert result.payload == {"inner": {"value": 1}, "tags": ["hooked"]}


def test_from_dict_applies_strict_extra_keys_to_nested_typed_dict():
    with pytest.raises(UnexpectedDataError) as exception_info:
        from_dict(NestedWrapper, {"payload": {"inner": {"value": 1}, "extra": 2}}, Config(strict=True))

    assert exception_info.value.field_path == "payload"
    assert exception_info.value.keys == {"extra"}


def test_from_dict_applies_strict_extra_keys_to_typed_dict_in_list():
    with pytest.raises(UnexpectedDataError) as exception_info:
        from_dict(ListWrapper, {"payloads": [{"value": 1}, {"value": 2, "extra": 3}]}, Config(strict=True))

    assert exception_info.value.field_path == "payloads[1]"
    assert exception_info.value.keys == {"extra"}


def test_from_dict_applies_strict_extra_keys_to_optional_typed_dict():
    with pytest.raises(UnexpectedDataError) as exception_info:
        from_dict(OptionalWrapper, {"payload": {"value": 1, "extra": 2}}, Config(strict=True))

    assert exception_info.value.field_path == "payload"
    assert exception_info.value.keys == {"extra"}


def test_from_dict_applies_strict_extra_keys_to_dataclass_in_typed_dict():
    @dataclass
    class Inner:
        value: int

    class Parent(TypedDict):
        inner: Inner

    @dataclass
    class Wrapper:
        payload: Parent

    with pytest.raises(UnexpectedDataError) as exception_info:
        from_dict(Wrapper, {"payload": {"inner": {"value": 1, "extra": 2}}}, Config(strict=True))

    assert exception_info.value.field_path == "payload.inner"
    assert exception_info.value.keys == {"extra"}


def test_from_dict_reports_missing_key_path_inside_typed_dict_list():
    with pytest.raises(MissingValueError) as exception_info:
        from_dict(ListWrapper, {"payloads": [{"value": 1}, {}]})

    assert exception_info.value.field_path == "payloads[1].value"


def test_from_dict_reports_wrong_key_path_inside_optional_typed_dict():
    with pytest.raises(WrongTypeError) as exception_info:
        from_dict(OptionalWrapper, {"payload": {"value": "bad"}})

    assert exception_info.value.field_path == "payload.value"
    assert exception_info.value.field_type is int


def test_from_dict_reports_wrong_key_path_inside_dataclass_in_typed_dict():
    @dataclass
    class Inner:
        value: int

    class Parent(TypedDict):
        inner: Inner

    @dataclass
    class Wrapper:
        payload: Parent

    with pytest.raises(WrongTypeError) as exception_info:
        from_dict(Wrapper, {"payload": {"inner": {"value": "bad"}}})

    assert exception_info.value.field_path == "payload.inner.value"


def test_from_dict_reports_non_mapping_typed_dict_path_inside_list():
    with pytest.raises(WrongTypeError) as exception_info:
        from_dict(ListWrapper, {"payloads": [1]})

    assert exception_info.value.field_path == "payloads[0]"
    assert exception_info.value.field_type is InnerTypedDict


def test_from_dict_resolves_typed_dict_forward_references():
    class ForwardTypedDict(TypedDict):
        inner: "ForwardInnerTypedDict"

    class ForwardInnerTypedDict(TypedDict):
        value: int

    @dataclass
    class Wrapper:
        payload: ForwardTypedDict

    result = from_dict(
        Wrapper,
        {"payload": {"inner": {"value": 1}}},
        Config(forward_references={"ForwardInnerTypedDict": ForwardInnerTypedDict}),
    )

    assert result.payload == {"inner": {"value": 1}}


def test_from_dict_creates_independent_containers_for_consecutive_inputs():
    first_payload = {"payload": {"inner": {"value": 1}, "tags": []}}
    second_payload = {"payload": {"inner": {"value": 2}, "tags": []}}

    first_result = from_dict(NestedWrapper, first_payload)
    second_result = from_dict(NestedWrapper, second_payload)
    first_result.payload["tags"].append("shared")

    assert first_result.payload is not second_result.payload
    assert first_result.payload["inner"] is not second_result.payload["inner"]
    assert first_result.payload["tags"] == ["shared"]
    assert second_result.payload["tags"] == []
    assert first_payload["payload"]["tags"] == []
