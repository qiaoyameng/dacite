from dataclasses import dataclass, field
from typing import Any, Generic, List, TypeVar

import pytest

from dacite import from_dict
from dacite.exceptions import TypeVarConstraintError, WrongTypeError


@dataclass
class X:
    a: str


@dataclass
class Y:
    b: str


T_bound = TypeVar("T_bound", bound=X)
T_constrained = TypeVar("T_constrained", int, str)
T_forward_bound = TypeVar("T_forward_bound", bound="X")
T_plain = TypeVar("T_plain")


@dataclass
class Bounded(Generic[T_bound]):
    x: T_bound


@dataclass
class Constrained(Generic[T_constrained]):
    x: T_constrained


@dataclass
class ForwardBounded(Generic[T_forward_bound]):
    x: T_forward_bound


def test_bound_satisfied():
    result = from_dict(data_class=Bounded[X], data={"x": {"a": "foo"}})

    assert result == Bounded(x=X(a="foo"))


def test_bound_violated():
    with pytest.raises(TypeVarConstraintError) as exc_info:
        from_dict(data_class=Bounded[Y], data={"x": {"b": "foo"}})

    error = exc_info.value
    assert error.type_var is T_bound
    assert error.value is Y
    assert '"T_bound"' in str(error)
    assert '"Y"' in str(error)
    assert 'bound "X"' in str(error)


def test_constraints_satisfied():
    assert from_dict(data_class=Constrained[int], data={"x": 1}) == Constrained(x=1)
    assert from_dict(data_class=Constrained[str], data={"x": "foo"}) == Constrained(x="foo")


def test_constraints_violated():
    with pytest.raises(TypeVarConstraintError) as exc_info:
        from_dict(data_class=Constrained[float], data={"x": 1.5})

    error = exc_info.value
    assert error.type_var is T_constrained
    assert error.value is float
    assert '"int"' in str(error) and '"str"' in str(error)


def test_forward_reference_bound_satisfied():
    result = from_dict(data_class=ForwardBounded[X], data={"x": {"a": "foo"}})

    assert result == ForwardBounded(x=X(a="foo"))


def test_forward_reference_bound_violated():
    with pytest.raises(TypeVarConstraintError):
        from_dict(data_class=ForwardBounded[Y], data={"x": {"b": "foo"}})


def test_multi_level_inheritance_passes_concrete_type():
    @dataclass
    class Middle(Bounded[X]):
        pass

    @dataclass
    class Leaf(Middle):
        pass

    result = from_dict(data_class=Leaf, data={"x": {"a": "foo"}})

    assert result == Leaf(x=X(a="foo"))

    with pytest.raises(WrongTypeError):
        from_dict(data_class=Leaf, data={"x": 1})


def test_multi_level_inheritance_distinguishes_same_name_type_vars():
    type_var_a = TypeVar("T")
    type_var_b = TypeVar("T")  # same name, different identity

    @dataclass
    class Base(Generic[type_var_a]):
        x: type_var_a

    @dataclass
    class Middle(Base[type_var_b], Generic[type_var_b]):
        pass

    @dataclass
    class Leaf(Middle[int]):
        pass

    result = from_dict(data_class=Leaf, data={"x": 1})

    assert result == Leaf(x=1)

    with pytest.raises(WrongTypeError):
        from_dict(data_class=Leaf, data={"x": "not an int"})


def test_bound_violated_through_inheritance():
    @dataclass
    class Legal(Bounded[X]):
        pass

    @dataclass
    class Illegal(Bounded[Y]):
        pass

    assert from_dict(data_class=Legal, data={"x": {"a": "foo"}}) == Legal(x=X(a="foo"))

    with pytest.raises(TypeVarConstraintError) as exc_info:
        from_dict(data_class=Illegal, data={"x": {"b": "foo"}})

    path = str(exc_info.value)
    assert "Illegal" in path
    assert "Bounded[" in path
    assert " -> " in path


def test_unspecialized_type_var_defaults_to_bound():
    result = from_dict(data_class=Bounded, data={"x": {"a": "foo"}})

    assert result == Bounded(x=X(a="foo"))

    with pytest.raises(WrongTypeError):
        from_dict(data_class=Bounded, data={"x": 1})


def test_unspecialized_type_var_defaults_to_union_of_constraints():
    assert from_dict(data_class=Constrained, data={"x": 1}) == Constrained(x=1)
    assert from_dict(data_class=Constrained, data={"x": "foo"}) == Constrained(x="foo")

    with pytest.raises(WrongTypeError):
        from_dict(data_class=Constrained, data={"x": 1.5})


def test_unspecialized_type_var_defaults_in_nested_container():
    @dataclass
    class Container(Generic[T_bound]):
        items: List[T_bound]

    result = from_dict(data_class=Container, data={"items": [{"a": "foo"}]})

    assert result == Container(items=[X(a="foo")])

    with pytest.raises(WrongTypeError):
        from_dict(data_class=Container, data={"items": [1]})


def test_unspecialized_forward_reference_bound():
    result = from_dict(data_class=ForwardBounded, data={"x": {"a": "foo"}})

    assert result == ForwardBounded(x=X(a="foo"))

    with pytest.raises(WrongTypeError):
        from_dict(data_class=ForwardBounded, data={"x": 1})


def test_unspecialized_plain_type_var_defaults_to_any():
    @dataclass
    class Plain(Generic[T_plain]):
        x: T_plain

    assert from_dict(data_class=Plain, data={"x": 1}) == Plain(x=1)
    assert from_dict(data_class=Plain, data={"x": "foo"}) == Plain(x="foo")


def test_cache_does_not_mask_illegal_specialization():
    assert from_dict(data_class=Bounded[X], data={"x": {"a": "foo"}}) == Bounded(x=X(a="foo"))

    with pytest.raises(TypeVarConstraintError):
        from_dict(data_class=Bounded[Y], data={"x": {"b": "foo"}})

    # a previously validated specialization keeps working ...
    assert from_dict(data_class=Bounded[X], data={"x": {"a": "foo"}}) == Bounded(x=X(a="foo"))

    # ... and an illegal one keeps failing
    with pytest.raises(TypeVarConstraintError):
        from_dict(data_class=Bounded[Y], data={"x": {"b": "foo"}})


def test_interleaved_subclasses_and_bound_violation():
    @dataclass
    class IntConstrained(Constrained[int]):
        pass

    @dataclass
    class StrConstrained(Constrained[str]):
        pass

    @dataclass
    class FloatConstrained(Constrained[float]):
        pass

    assert from_dict(data_class=IntConstrained, data={"x": 1}) == IntConstrained(x=1)
    assert from_dict(data_class=StrConstrained, data={"x": "foo"}) == StrConstrained(x="foo")
    assert from_dict(data_class=IntConstrained, data={"x": 2}) == IntConstrained(x=2)
    assert from_dict(data_class=StrConstrained, data={"x": "bar"}) == StrConstrained(x="bar")

    with pytest.raises(TypeVarConstraintError):
        from_dict(data_class=FloatConstrained, data={"x": 1.5})


def test_validation_happens_before_default_factory_and_post_init():
    calls = []

    @dataclass
    class WithDefaults(Generic[T_bound]):
        x: T_bound
        y: int = field(default_factory=lambda: calls.append("default_factory") or 1)

        def __post_init__(self):
            calls.append("post_init")

    with pytest.raises(TypeVarConstraintError):
        from_dict(data_class=WithDefaults[Y], data={"x": {"b": "foo"}})

    assert calls == []


def test_specialization_with_any_is_allowed():
    assert from_dict(data_class=Bounded[Any], data={"x": 1}) == Bounded(x=1)
