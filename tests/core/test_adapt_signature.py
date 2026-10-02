import sys
from dataclasses import dataclass, InitVar, field

import pytest

from dacite import Config, SignatureBindingError, from_dict
from dacite.core import _bind_init_values
from tests.common import init_var_type_support


def test_from_dict_with_adapt_signature_and_inheritance_reordered_fields():
    calls = []

    @dataclass(init=False)
    class Base:
        z: int
        a: int

        def __init__(self, a, z, *, base_only=None):
            self.a = a
            self.z = z
            self.base_only = base_only

    @dataclass(init=False)
    class Child(Base):
        c: int
        b: int

        def __init__(self, a, b, c, z, *, flag=False, **extra):
            calls.append({"args": (a, b, c, z), "kwargs": {"flag": flag, "extra": extra}})
            super().__init__(a=a, z=z, base_only=extra.get("base_only"))
            self.b = b
            self.c = c
            self.flag = flag

    result = from_dict(
        Child,
        {"z": 1, "a": 2, "c": 3, "b": 4, "base_only": 9},
        config=Config(adapt_signature=True),
    )

    assert calls == [{"args": (2, 4, 3, 1), "kwargs": {"flag": False, "extra": {"base_only": 9}}}]
    assert (result.a, result.b, result.c, result.z) == (2, 4, 3, 1)
    assert result.base_only == 9


def test_bind_init_values_uses_constructor_parameter_order():
    @dataclass(init=False)
    class X:
        z: int
        a: int
        b: int

        def __init__(self, a, b, z):
            ...

    args, kwargs = _bind_init_values(X, {"z": 1, "a": 2, "b": 3})

    assert args == [2, 3, 1]
    assert kwargs == {}


def test_bind_init_values_with_defaulted_non_field_parameter():
    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a, b=10):
            ...

    args, kwargs = _bind_init_values(X, {"a": 1})

    assert args == [1]
    assert kwargs == {}


def test_bind_init_values_with_keyword_only_parameters():
    @dataclass(init=False)
    class X:
        a: int
        mode: str

        def __init__(self, a, *, mode):
            ...

    args, kwargs = _bind_init_values(X, {"a": 1, "mode": "fast"})

    assert args == [1]
    assert kwargs == {"mode": "fast"}


def test_bind_init_values_rejects_required_unknown_keyword_only_parameter():
    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a, *, mode):
            ...

    with pytest.raises(SignatureBindingError) as exc_info:
        _bind_init_values(X, {"a": 1})

    assert "mode" in str(exc_info.value)


def test_bind_init_values_rejects_required_unknown_parameter():
    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a, b):
            ...

    with pytest.raises(SignatureBindingError) as exc_info:
        _bind_init_values(X, {"a": 1})

    assert "b" in str(exc_info.value)


def test_bind_init_values_rejects_var_positional():
    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a, *args):
            ...

    with pytest.raises(SignatureBindingError) as exc_info:
        _bind_init_values(X, {"a": 1})

    assert "args" in str(exc_info.value)


def test_bind_init_values_rejects_fields_without_matching_parameters():
    @dataclass(init=False)
    class X:
        a: int
        extra: int

        def __init__(self, a):
            ...

    with pytest.raises(SignatureBindingError) as exc_info:
        _bind_init_values(X, {"a": 1, "extra": 2})

    assert "extra" in str(exc_info.value)


def test_bind_init_values_maps_leftovers_to_var_keyword():
    @dataclass(init=False)
    class X:
        a: int
        extra: int

        def __init__(self, a, **kwargs):
            ...

    args, kwargs = _bind_init_values(X, {"a": 1, "extra": 2})

    assert args == [1]
    assert kwargs == {"extra": 2}


@init_var_type_support
def test_from_dict_with_adapt_signature_maps_init_var():
    calls = []

    @dataclass(init=False)
    class X:
        a: InitVar[int]
        b: int = field(init=False)

        def __init__(self, a):
            calls.append(a)
            self.b = 2 * a

    result = from_dict(X, {"a": 3}, config=Config(adapt_signature=True))

    assert calls == [3]
    assert result.b == 6


def test_from_dict_with_adapt_signature_converts_values_before_binding():
    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a):
            self.a = a

    result = from_dict(X, {"a": "5"}, config=Config(adapt_signature=True, cast=[int]))

    assert result.a == 5


def test_from_dict_with_adapt_signature_for_nested_dataclasses():
    calls = []

    @dataclass(init=False)
    class Inner:
        value: int

        def __init__(self, value):
            calls.append(("inner", value))
            self.value = value

    @dataclass(init=False)
    class Outer:
        inner: Inner
        name: str

        def __init__(self, name, inner):
            calls.append(("outer", name, inner))
            self.name = name
            self.inner = inner

    result = from_dict(
        Outer,
        {"name": "x", "inner": {"value": 7}},
        config=Config(adapt_signature=True),
    )

    assert len(calls) == 2
    assert calls[0] == ("inner", 7)
    assert calls[1][:2] == ("outer", "x")
    assert type(calls[1][2]).__name__ == "Inner"
    assert calls[1][2].value == 7
    assert result.inner.value == 7


def test_from_dict_with_constructor_type_error_is_not_retried():
    class BoomError(TypeError):
        pass

    calls = []

    @dataclass(init=False)
    class Boom:
        x: int

        def __init__(self, x):
            calls.append(x)
            raise BoomError("deliberate failure inside constructor")

    with pytest.raises(BoomError, match="deliberate failure inside constructor"):
        from_dict(Boom, {"x": 1}, config=Config(adapt_signature=True))

    assert calls == [1]


@pytest.mark.skipif(sys.version_info < (3, 8), reason="positional-only parameters require Python 3.8")
def test_bind_init_values_with_positional_only_parameters():
    namespace = {}
    exec(
        """
def __init__(self, a, b, /, c=3):
    ...
""",
        namespace,
    )

    @dataclass(init=False)
    class X:
        a: int
        b: int
        c: int = 0

    X.__init__ = namespace["__init__"]

    args, kwargs = _bind_init_values(X, {"a": 1, "b": 2, "c": 4})

    assert args == [1, 2, 4]
    assert kwargs == {}
