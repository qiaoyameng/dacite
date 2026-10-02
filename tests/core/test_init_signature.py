from dataclasses import dataclass, InitVar

import pytest

from dacite import from_dict, Config
from dacite.exceptions import InitSignatureError


def test_adapt_init_signature_with_inherited_fields_and_different_parameter_order():
    calls = []

    @dataclass
    class Base:
        a: int
        b: str

    @dataclass(init=False)
    class Child(Base):
        c: int

        def __init__(self, c: int, a: int, /, b: str = "default", *, scale: int = 1):
            calls.append(((c, a), b, scale))
            self.a = a * scale
            self.b = b
            self.c = c

    result = from_dict(data_class=Child, data={"a": 2, "b": "x", "c": 3}, config=Config(adapt_init_signature=True))

    assert calls == [((3, 2), "x", 1)]
    assert result.a == 2
    assert result.b == "x"
    assert result.c == 3


def test_adapt_init_signature_converts_fields_before_mapping():
    calls = []

    @dataclass(init=False)
    class Point:
        x: int
        y: int

        def __init__(self, x, y, /):
            calls.append((x, y))
            self.x = x
            self.y = y

    result = from_dict(
        data_class=Point,
        data={"x": "1", "y": "2"},
        config=Config(adapt_init_signature=True, type_hooks={int: int}),
    )

    assert calls == [(1, 2)]
    assert result.x == 1
    assert result.y == 2


def test_adapt_init_signature_with_init_var():
    @dataclass(init=False)
    class X:
        value: int
        scale: InitVar[int] = 2

        def __init__(self, value, /, *, scale):
            self.value = value * scale

    result = from_dict(data_class=X, data={"value": 3, "scale": 4}, config=Config(adapt_init_signature=True))

    assert result.value == 12


def test_adapt_init_signature_with_init_var_default():
    @dataclass(init=False)
    class X:
        value: int
        scale: InitVar[int] = 2

        def __init__(self, value, /, *, scale):
            self.value = value * scale

    result = from_dict(data_class=X, data={"value": 3}, config=Config(adapt_init_signature=True))

    assert result.value == 6


def test_adapt_init_signature_passes_unmatched_fields_to_variadic_keyword_parameter():
    @dataclass(init=False)
    class X:
        a: int
        b: str

        def __init__(self, a, /, **kwargs):
            self.a = a
            self.b = kwargs["b"]

    result = from_dict(data_class=X, data={"a": 1, "b": "x"}, config=Config(adapt_init_signature=True))

    assert result.a == 1
    assert result.b == "x"


def test_adapt_init_signature_rejects_variadic_positional_parameter_before_call():
    calls = []

    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a, *args):
            calls.append(a)
            self.a = a

    with pytest.raises(InitSignatureError) as exc_info:
        from_dict(data_class=X, data={"a": 1}, config=Config(adapt_init_signature=True))

    assert "*args" in str(exc_info.value)
    assert calls == []


def test_adapt_init_signature_rejects_required_parameter_not_matching_a_field_before_call():
    calls = []

    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a, b):
            calls.append((a, b))
            self.a = a

    with pytest.raises(InitSignatureError) as exc_info:
        from_dict(data_class=X, data={"a": 1}, config=Config(adapt_init_signature=True))

    assert '"b"' in str(exc_info.value)
    assert calls == []


def test_adapt_init_signature_rejects_fields_not_matching_any_parameter_before_call():
    calls = []

    @dataclass(init=False)
    class X:
        a: int
        b: int

        def __init__(self, a):
            calls.append(a)
            self.a = a

    with pytest.raises(InitSignatureError) as exc_info:
        from_dict(data_class=X, data={"a": 1, "b": 2}, config=Config(adapt_init_signature=True))

    assert '"b"' in str(exc_info.value)
    assert calls == []


def test_adapt_init_signature_propagates_type_error_raised_by_init_without_retry():
    calls = []

    @dataclass(init=False)
    class X:
        a: int

        def __init__(self, a):
            calls.append(a)
            raise TypeError("broken inside init")

    with pytest.raises(TypeError, match="broken inside init"):
        from_dict(data_class=X, data={"a": 1}, config=Config(adapt_init_signature=True))

    assert calls == [1]
