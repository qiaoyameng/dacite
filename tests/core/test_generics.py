from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar, get_type_hints

import pytest

from dacite import Config, TypeVarBoundError, from_dict
from dacite.generics import get_concrete_type_hints

from tests.common import pep_604_support

T = TypeVar("T")
U = TypeVar("U")
N = TypeVar("N", bound=int)
S = TypeVar("S", int, str)
fwd_t = TypeVar("FwdT", bound="ForwardBase")


@dataclass
class ForwardBase:
    pass


@dataclass
class ForwardDerived(ForwardBase):
    pass


@dataclass
class X:
    a: str


@dataclass
class A(Generic[T, U]):
    x: T
    y: List[U]


def test_multi_generic():
    data = {
        "x": {
            "a": "foo",
        },
        "y": [1, 2, 3],
    }

    result = from_dict(data_class=A[X, int], data=data)

    assert result == A(x=X(a="foo"), y=[1, 2, 3])

    # assert that the typing object hasn't been modified in-place
    hints = get_type_hints(A)
    assert hints["x"].__class__ is TypeVar


def test_generic_parent():
    @dataclass
    class B(A[X, int]):
        z: str

    data = {
        "x": {
            "a": "foo",
        },
        "y": [1, 2, 3],
        "z": "bar",
    }

    result = from_dict(data_class=B, data=data)

    assert result == B(x=X(a="foo"), y=[1, 2, 3], z="bar")


def test_generic_field():
    @dataclass
    class C:
        z: A[X, int]

    data = {
        "z": {
            "x": {
                "a": "foo",
            },
            "y": [1, 2, 3],
        }
    }

    result = from_dict(data_class=C, data=data)

    assert result == C(z=A(x=X(a="foo"), y=[1, 2, 3]))


@pep_604_support
def test_pep604():
    @dataclass
    class D(Generic[T]):
        x: list[T] | int | None  # pylint: disable=unsupported-binary-operation

    data = {"x": [{"a": "foo"}, {"a": "bar"}]}

    result = from_dict(data_class=D[X], data=data)

    assert result == D[X](x=[X(a="foo"), X(a="bar")])


@dataclass
class Bounded(Generic[N]):
    x: N
    y: List[N]


def test_bound_satisfied():
    result = from_dict(data_class=Bounded[int], data={"x": 1, "y": [2, 3]})

    assert result == Bounded(x=1, y=[2, 3])


def test_bound_violated():
    with pytest.raises(TypeVarBoundError) as exc_info:
        from_dict(data_class=Bounded[str], data={"x": "a", "y": ["b"]})

    message = str(exc_info.value)
    assert '"N"' in message  # the declared type variable
    assert '"str"' in message  # the actual type
    assert "Bounded[str]" in message  # the inheritance path


def test_bound_violated_inside_nested_container():
    with pytest.raises(TypeVarBoundError):
        from_dict(data_class=Bounded[str], data={"x": 1, "y": [2]})


def test_constraints_satisfied():
    @dataclass
    class Constrained(Generic[S]):
        x: S

    assert from_dict(data_class=Constrained[int], data={"x": 1}) == Constrained(x=1)
    assert from_dict(data_class=Constrained[str], data={"x": "a"}) == Constrained(x="a")


def test_constraints_violated():
    @dataclass
    class Constrained(Generic[S]):
        x: S

    with pytest.raises(TypeVarBoundError) as exc_info:
        from_dict(data_class=Constrained[float], data={"x": 1.0})

    message = str(exc_info.value)
    assert '"S"' in message
    assert '"float"' in message
    assert '"int"' in message and '"str"' in message  # the declared constraints


def test_bound_violated_through_multi_level_inheritance():
    @dataclass
    class Middle(Bounded[int]):
        pass

    @dataclass
    class Leaf(Middle):
        pass

    assert from_dict(data_class=Leaf, data={"x": 1, "y": [2]}) == Leaf(x=1, y=[2])

    @dataclass
    class BadMiddle(Bounded[str]):
        pass

    @dataclass
    class BadLeaf(BadMiddle):
        pass

    with pytest.raises(TypeVarBoundError) as exc_info:
        from_dict(data_class=BadLeaf, data={"x": "a", "y": ["b"]})

    message = str(exc_info.value)
    assert "BadLeaf -> BadMiddle -> Bounded[str]" in message


def test_same_named_type_vars_with_different_identity():
    base_t = TypeVar("T", bound=int)
    sub_t = TypeVar("T")

    @dataclass
    class Base(Generic[base_t]):  # pylint: disable=non-typename-type-var
        x: base_t

    @dataclass
    class Sub(Base[sub_t], Generic[sub_t]):  # pylint: disable=non-typename-type-var
        pass

    assert from_dict(data_class=Sub[int], data={"x": 1}) == Sub(x=1)

    with pytest.raises(TypeVarBoundError) as exc_info:
        from_dict(data_class=Sub[str], data={"x": "a"})

    assert "Sub -> Base[int]" in str(exc_info.value) or "Base[str]" in str(exc_info.value)


def test_unspecialized_type_var_falls_back_to_type_var_not_any():
    hints = get_concrete_type_hints(Bounded)

    assert hints["x"] is N
    assert hints["x"] is not None

    # documented fallback: an unspecialized type var remains a TypeVar,
    # so with check_types=False from_dict still works (without type checking)
    result = from_dict(data_class=Bounded, data={"x": "a", "y": ["b"]}, config=Config(check_types=False))
    assert result == Bounded(x="a", y=["b"])


def test_bound_declared_as_forward_reference():
    @dataclass
    class Holder(Generic[fwd_t]):
        x: fwd_t

    assert from_dict(data_class=Holder[ForwardDerived], data={"x": {}}) == Holder(x=ForwardDerived())

    with pytest.raises(TypeVarBoundError):
        from_dict(data_class=Holder[int], data={"x": 1})


def test_bound_enforced_for_string_forward_reference_field():
    @dataclass
    class FwdRef(Generic[N]):
        x: "N"
        y: Optional["Bounded[int]"]

    assert from_dict(data_class=FwdRef[int], data={"x": 1, "y": None}) == FwdRef(x=1, y=None)

    with pytest.raises(TypeVarBoundError):
        from_dict(data_class=FwdRef[str], data={"x": "a", "y": None})


def test_interleaved_subclasses_and_cache_do_not_mask_violations():
    calls = []

    @dataclass
    class IntChild(Bounded[int]):
        def __post_init__(self):
            calls.append("int")

    @dataclass
    class StrField:
        s: str

    unbounded_t = TypeVar("UnboundedT")

    @dataclass
    class OtherBase(Generic[unbounded_t]):
        x: unbounded_t

    @dataclass
    class StrChild(OtherBase[str]):
        def __post_init__(self):
            calls.append("str")

    # interleave construction of both subclasses - each has its own cache entry
    for _ in range(2):
        int_child = from_dict(data_class=IntChild, data={"x": 1, "y": [2]})
        str_child = from_dict(data_class=StrChild, data={"x": "a"})
        assert (int_child.x, int_child.y) == (1, [2])
        assert str_child.x == "a"

    assert calls == ["int", "str", "int", "str"]

    # a third subclass violating the bound still fails, even though
    # legal specializations of the same generic base were cached before
    @dataclass
    class BadChild(Bounded[str]):
        z: List[str] = field(default_factory=lambda: calls.append("factory"))

        def __post_init__(self):
            calls.append("bad")

    with pytest.raises(TypeVarBoundError):
        from_dict(data_class=BadChild, data={"x": "a", "y": ["b"]})

    # neither the default factory nor __post_init__ were executed
    assert calls == ["int", "str", "int", "str"]

    # and legal specializations still work afterwards
    int_child = from_dict(data_class=IntChild, data={"x": 1, "y": [2]})
    assert (int_child.x, int_child.y) == (1, [2])


def test_type_var_bound_error_is_raised_again_for_cached_legal_specialization():
    # a cached legal specialization must not let a later illegal one slip through
    assert from_dict(data_class=Bounded[int], data={"x": 1, "y": [2]}) == Bounded(x=1, y=[2])

    with pytest.raises(TypeVarBoundError):
        from_dict(data_class=Bounded[bytes], data={"x": b"a", "y": [b"b"]})

    with pytest.raises(TypeVarBoundError):
        from_dict(data_class=Bounded[bytes], data={"x": b"a", "y": [b"b"]})
