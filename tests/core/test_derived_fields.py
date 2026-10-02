import sys
from dataclasses import dataclass, field
from typing import ClassVar, List

import pytest

from dacite import (
    from_dict,
    Config,
    DaciteError,
    DerivedFieldsPolicy,
    DerivedFieldInputError,
    DerivedFieldMismatchError,
)

slots_support = pytest.mark.skipif(sys.version_info < (3, 10), reason="requires Python 3.10")


def make_order_classes():
    @dataclass
    class LineItem:
        name: str
        price: float

    @dataclass
    class Order:
        items: List[LineItem]
        total: float = field(init=False)
        constructions: ClassVar[int] = 0

        def __post_init__(self):
            type(self).constructions += 1
            self.total = sum(item.price for item in self.items)

    return LineItem, Order


VALIDATE = Config(derived_fields_policy=DerivedFieldsPolicy.VALIDATE)
REJECT = Config(derived_fields_policy=DerivedFieldsPolicy.REJECT)


def test_validate_with_omitted_derived_field():
    _, Order = make_order_classes()

    result = from_dict(Order, {"items": [{"name": "a", "price": 2.0}, {"name": "b", "price": 3.0}]}, VALIDATE)

    assert result.total == 5.0
    assert Order.constructions == 1


def test_validate_with_correct_derived_field():
    _, Order = make_order_classes()

    result = from_dict(
        Order, {"items": [{"name": "a", "price": 2.0}, {"name": "b", "price": 3.0}], "total": 5.0}, VALIDATE
    )

    assert result.total == 5.0
    assert Order.constructions == 1


def test_validate_with_forged_derived_field():
    _, Order = make_order_classes()

    with pytest.raises(DerivedFieldMismatchError) as exc_info:
        from_dict(Order, {"items": [{"name": "a", "price": 2.0}], "total": 100.0}, VALIDATE)

    assert exc_info.value.field_path == "total"
    assert exc_info.value.value == 100.0
    assert exc_info.value.computed == 2.0
    assert Order.constructions == 1


def test_validate_does_not_mutate_input_data():
    _, Order = make_order_classes()
    data = {"items": [{"name": "a", "price": 2.0}], "total": 2.0}
    snapshot = {"items": [{"name": "a", "price": 2.0}], "total": 2.0}

    from_dict(Order, data, VALIDATE)

    assert data == snapshot


def test_validate_applies_type_hooks_before_comparison():
    @dataclass
    class X:
        s: str
        code: str = field(init=False)

        def __post_init__(self):
            self.code = self.s.lower()

    config = Config(derived_fields_policy=DerivedFieldsPolicy.VALIDATE, type_hooks={str: str.lower})

    result = from_dict(X, {"s": "AB", "code": "ab"}, config)

    assert result.code == "ab"


def test_validate_nested_mismatch_points_to_derived_field():
    _, Order = make_order_classes()

    @dataclass
    class Cart:
        order: Order

    data = {"order": {"items": [{"name": "a", "price": 2.0}], "total": 100.0}}

    with pytest.raises(DerivedFieldMismatchError) as exc_info:
        from_dict(Cart, data, VALIDATE)

    assert exc_info.value.field_path == "order.total"


def test_validate_with_frozen_dataclass():
    @dataclass(frozen=True)
    class X:
        a: int
        b: int = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "b", self.a * 2)

    assert from_dict(X, {"a": 2, "b": 4}, VALIDATE).b == 4
    with pytest.raises(DerivedFieldMismatchError):
        from_dict(X, {"a": 2, "b": 5}, VALIDATE)


@slots_support
def test_validate_with_slots_dataclass():
    @dataclass(slots=True)
    class X:
        a: int
        b: int = field(init=False)

        def __post_init__(self):
            self.b = self.a * 2

    assert from_dict(X, {"a": 2, "b": 4}, VALIDATE).b == 4
    with pytest.raises(DerivedFieldMismatchError):
        from_dict(X, {"a": 2, "b": 5}, VALIDATE)


def test_validate_propagates_post_init_exceptions_unchanged():
    @dataclass
    class X:
        a: int
        b: int = field(init=False)

        def __post_init__(self):
            raise ValueError("boom")

    with pytest.raises(ValueError) as exc_info:
        from_dict(X, {"a": 1, "b": 2}, VALIDATE)

    assert not isinstance(exc_info.value, DaciteError)


def test_reject_with_provided_derived_field():
    _, Order = make_order_classes()

    with pytest.raises(DerivedFieldInputError) as exc_info:
        from_dict(Order, {"items": [{"name": "a", "price": 2.0}], "total": 2.0}, REJECT)

    assert exc_info.value.field_path == "total"
    assert Order.constructions == 0


def test_reject_with_omitted_derived_field():
    _, Order = make_order_classes()

    result = from_dict(Order, {"items": [{"name": "a", "price": 2.0}]}, REJECT)

    assert result.total == 2.0
    assert Order.constructions == 1


def test_reject_nested_points_to_derived_field():
    _, Order = make_order_classes()

    @dataclass
    class Cart:
        order: Order

    data = {"order": {"items": [{"name": "a", "price": 2.0}], "total": 2.0}}

    with pytest.raises(DerivedFieldInputError) as exc_info:
        from_dict(Cart, data, REJECT)

    assert exc_info.value.field_path == "order.total"


def test_default_policy_keeps_existing_behavior():
    _, Order = make_order_classes()

    result = from_dict(Order, {"items": [{"name": "a", "price": 2.0}], "total": 100.0})

    assert result.total == 100.0
    assert Order.constructions == 1
