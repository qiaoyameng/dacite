from dataclasses import dataclass
from typing import Annotated, Generic, List, Optional, TypeVar

import pytest

from dacite import (
    AfterTypeConstructionProcessor,
    AnnotationProcessingError,
    AnnotationStage,
    BeforeTypeConstructionProcessor,
    Config,
    UnknownAnnotationError,
    from_dict,
)


@dataclass(frozen=True)
class Trim:
    field: str


@dataclass(frozen=True)
class Preserve:
    field: str


@dataclass(frozen=True)
class MinLength:
    field: str
    length: int


@dataclass(frozen=True)
class ListLength:
    field: str


class TrimProcessor(BeforeTypeConstructionProcessor):
    calls = []
    events = []

    def process(self, value):
        event = ("trim", self.metadata.field, value)
        self.__class__.calls.append(event)
        TrimProcessor.events.append(event)
        return value.strip()


class PreserveProcessor(BeforeTypeConstructionProcessor):
    calls = []

    def process(self, value):
        event = ("preserve", self.metadata.field, value)
        self.__class__.calls.append(event)
        TrimProcessor.events.append(event)
        return value


class MinLengthProcessor(AfterTypeConstructionProcessor):
    calls = []
    instances = []

    def __init__(self, metadata):
        super().__init__(metadata)
        self.__class__.instances.append(self)

    def process(self, value):
        event = ("minimum_length", self.metadata.field, value)
        self.__class__.calls.append(event)
        TrimProcessor.events.append(event)
        if len(value) < self.metadata.length:
            raise ValueError("too short")
        return value


class ListLengthProcessor(AfterTypeConstructionProcessor):
    calls = []

    def process(self, value):
        event = ("list_length", len(value))
        self.__class__.calls.append(event)
        TrimProcessor.events.append(event)
        return value


@dataclass
class Item:
    name: Annotated[str, Trim("name"), MinLength("name", 1)]


@dataclass
class Form:
    trimmed: Annotated[str, Trim("trimmed"), MinLength("trimmed", 3)]
    preserved: Annotated[str, Preserve("preserved")]
    items: Annotated[List[Annotated[str, Trim("items.name")]], ListLength("items")]


@pytest.fixture(autouse=True)
def clear_processor_calls():
    TrimProcessor.calls = []
    PreserveProcessor.calls = []
    MinLengthProcessor.calls = []
    ListLengthProcessor.calls = []
    TrimProcessor.events = []
    MinLengthProcessor.instances = []
    yield


def annotation_config():
    return Config(
        annotation_processors={
            Trim: TrimProcessor,
            Preserve: PreserveProcessor,
            MinLength: MinLengthProcessor,
            ListLength: ListLengthProcessor,
        }
    )


def test_annotation_processors_run_before_and_after_type_construction_with_nested_list():
    hook_calls = []

    def list_hook(value):
        hook_calls.append("str")
        TrimProcessor.events.append(("list_hook", len(value)))
        return value

    def integer_list_hook(value):
        hook_calls.append("int")
        return value

    result = from_dict(
        Form,
        {
            "trimmed": " abc ",
            "preserved": " cd ",
            "items": [" x ", " y"],
        },
        Config(
            annotation_processors={
                Trim: TrimProcessor,
                Preserve: PreserveProcessor,
                MinLength: MinLengthProcessor,
                ListLength: ListLengthProcessor,
            },
            type_hooks={List[str]: list_hook, List[int]: integer_list_hook},
        ),
    )

    assert result == Form(trimmed="abc", preserved=" cd ", items=["x", "y"])
    assert TrimProcessor.events == [
        ("trim", "trimmed", " abc "),
        ("minimum_length", "trimmed", "abc"),
        ("preserve", "preserved", " cd "),
        ("list_hook", 2),
        ("trim", "items.name", " x "),
        ("trim", "items.name", " y"),
        ("list_length", 2),
    ]
    assert MinLengthProcessor.calls == [
        ("minimum_length", "trimmed", "abc"),
    ]
    assert ListLengthProcessor.calls == [("list_length", 2)]
    assert hook_calls == ["str"]


def test_post_construction_validation_uses_constructed_value_and_keeps_path_and_stage():
    with pytest.raises(AnnotationProcessingError) as exception_info:
        from_dict(
            Form,
            {"trimmed": " x ", "preserved": "ok", "items": []},
            annotation_config(),
        )

    assert exception_info.value.field_path == "trimmed"
    assert exception_info.value.stage == AnnotationStage.AFTER_TYPE_CONSTRUCTION
    assert exception_info.value.value == "x"
    assert isinstance(exception_info.value.original_error, ValueError)


def test_global_type_hook_runs_after_before_construction_processors():
    @dataclass
    class Simple:
        trimmed: Annotated[str, Trim("trimmed")]

    result = from_dict(
        Simple,
        {"trimmed": " abc "},
        Config(
            annotation_processors={Trim: TrimProcessor},
            type_hooks={str: str.upper},
        ),
    )

    assert result == Simple(trimmed="ABC")
    assert TrimProcessor.calls == [("trim", "trimmed", " abc ")]


def test_failed_validation_does_not_leave_temporary_state_for_next_valid_conversion():
    with pytest.raises(AnnotationProcessingError):
        from_dict(
            Item,
            {"name": "   "},
            annotation_config(),
        )

    assert MinLengthProcessor.calls == [("minimum_length", "name", "")]
    failed_processor = MinLengthProcessor.instances[0]

    result = from_dict(
        Item,
        {"name": " ok "},
        annotation_config(),
    )

    assert result == Item(name="ok")
    assert MinLengthProcessor.calls == [
        ("minimum_length", "name", ""),
        ("minimum_length", "name", "ok"),
    ]
    assert MinLengthProcessor.instances[1] is not failed_processor


def test_nested_collection_item_validation_keeps_indexed_path():
    @dataclass
    class Container:
        items: List[Annotated[str, MinLength("items", 2)]]

    with pytest.raises(AnnotationProcessingError) as exception_info:
        from_dict(
            Container,
            {"items": ["ok", "x"]},
            annotation_config(),
        )

    assert exception_info.value.field_path == "items.1"


def test_same_base_type_with_different_annotations_uses_each_annotation_metadata():
    @dataclass
    class DifferentRules:
        strict_name: Annotated[str, MinLength("strict_name", 5)]
        short_code: Annotated[str, MinLength("short_code", 2)]

    result = from_dict(
        DifferentRules,
        {"strict_name": "valid", "short_code": "ok"},
        annotation_config(),
    )

    assert result == DifferentRules(strict_name="valid", short_code="ok")
    assert MinLengthProcessor.calls == [
        ("minimum_length", "strict_name", "valid"),
        ("minimum_length", "short_code", "ok"),
    ]


def test_unknown_annotation_is_retained_by_default_and_rejected_in_strict_mode():
    @dataclass
    class Unknown:
        value: Annotated[str, object()]

    assert from_dict(Unknown, {"value": "ok"}) == Unknown(value="ok")

    with pytest.raises(UnknownAnnotationError) as exception_info:
        from_dict(Unknown, {"value": "ok"}, Config(strict=True))

    assert exception_info.value.stage == AnnotationStage.BEFORE_TYPE_CONSTRUCTION


def test_annotated_optional_short_circuits_none_before_processors():
    class RecordOptional(AfterTypeConstructionProcessor):
        calls = []

        def process(self, value):
            self.__class__.calls.append(value)
            return value

    RecordOptional.calls = []

    @dataclass
    class OptionalForm:
        value: Optional[Annotated[str, ListLength("value")]]

    result = from_dict(
        OptionalForm,
        {"value": None},
        Config(annotation_processors={ListLength: RecordOptional}),
    )

    assert result == OptionalForm(value=None)
    assert RecordOptional.calls == []


def test_annotation_metadata_is_preserved_for_generic_type_vars():
    type_var = TypeVar("type_var")

    @dataclass
    class GenericForm(Generic[type_var]):
        value: Annotated[type_var, Trim("generic_value")]

    result = from_dict(
        GenericForm[str],
        {"value": " ok "},
        Config(annotation_processors={Trim: TrimProcessor}),
    )

    assert result == GenericForm(value="ok")
    assert TrimProcessor.calls == [("trim", "generic_value", " ok ")]


def test_annotated_optional_processors_run_once_for_each_optional_form():
    @dataclass
    class BothOptionalForms:
        inner: Optional[Annotated[str, ListLength("inner")]]
        outer: Annotated[Optional[str], ListLength("outer")]

    from_dict(
        BothOptionalForms,
        {"inner": "a", "outer": "b"},
        Config(annotation_processors={ListLength: ListLengthProcessor}),
    )

    assert ListLengthProcessor.calls == [("list_length", 1), ("list_length", 1)]


def test_annotated_optional_none_short_circuits_outer_processors():
    @dataclass
    class OuterOptionalForm:
        value: Annotated[Optional[str], Trim("value"), ListLength("value")]

    result = from_dict(
        OuterOptionalForm,
        {"value": None},
        Config(
            annotation_processors={
                ListLength: ListLengthProcessor,
                Trim: TrimProcessor,
            }
        ),
    )

    assert result == OuterOptionalForm(value=None)
    assert ListLengthProcessor.calls == []
    assert TrimProcessor.calls == []


def test_post_construction_processor_runs_after_base_type_cast():
    seen = []

    class PositiveInteger(AfterTypeConstructionProcessor):
        def process(self, value):
            seen.append(value)
            return value

    @dataclass(frozen=True)
    class Positive:
        pass

    @dataclass
    class CastForm:
        count: Annotated[int, Positive()]

    result = from_dict(
        CastForm,
        {"count": "42"},
        Config(annotation_processors={Positive: PositiveInteger}, cast=[int]),
    )

    assert result == CastForm(count=42)
    assert seen == [42]
