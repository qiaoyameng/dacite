import sys
from dataclasses import dataclass, field
from typing import List

import pytest

from dacite import (
    Config,
    DerivedFieldPolicy,
    DerivedFieldInputError,
    DerivedFieldMismatchError,
    WrongTypeError,
    from_dict,
)


@dataclass
class Item:
    amount: int
    total: int = field(init=False)
    post_init_count = 0

    def __post_init__(self) -> None:
        Item.post_init_count += 1
        self.total = self.amount


@dataclass
class Order:
    items: List[Item]
    amounts: List[int] = field(init=False)
    total: int = field(init=False)
    post_init_count = 0

    def __post_init__(self) -> None:
        Order.post_init_count += 1
        self.amounts = [item.amount for item in self.items]
        self.total = sum(self.amounts)


@dataclass
class Summary:
    value: int
    post_init_count = 0

    def __post_init__(self) -> None:
        Summary.post_init_count += 1


@dataclass
class OrderWithSummaries:
    items: List[Item]
    summaries: List[Summary] = field(init=False)
    post_init_count = 0

    def __post_init__(self) -> None:
        OrderWithSummaries.post_init_count += 1
        self.summaries = [Summary(value=item.amount) for item in self.items]


@dataclass
class OrderWithSummary:
    items: List[Item]
    summary: Summary = field(init=False)
    total: int = field(init=False)
    post_init_count = 0

    def __post_init__(self) -> None:
        OrderWithSummary.post_init_count += 1
        self.total = sum(item.amount for item in self.items)
        self.summary = Summary(value=self.total)


@pytest.fixture(autouse=True)
def reset_post_init_counts():
    Item.post_init_count = 0
    Order.post_init_count = 0
    OrderWithSummary.post_init_count = 0
    OrderWithSummaries.post_init_count = 0
    Summary.post_init_count = 0


def test_assert_policy_allows_omitting_derived_fields_and_constructs_once():
    result = from_dict(
        Order,
        {"items": [{"amount": 2}, {"amount": 3}]},
        Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
    )

    assert [item.amount for item in result.items] == [2, 3]
    assert result.amounts == [2, 3]
    assert result.total == 5
    assert Item.post_init_count == 2
    assert Order.post_init_count == 1


def test_assert_policy_accepts_correct_derived_values_and_nested_values():
    data = {
        "items": [{"amount": 2, "total": 2}, {"amount": 3, "total": 3}],
        "amounts": [2, 3],
        "total": 5,
    }

    result = from_dict(
        Order,
        data,
        Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
    )

    assert result.total == 5
    assert Item.post_init_count == 2
    assert Order.post_init_count == 1
    assert data == {
        "items": [{"amount": 2, "total": 2}, {"amount": 3, "total": 3}],
        "amounts": [2, 3],
        "total": 5,
    }


def test_assert_policy_rejects_forged_derived_total_after_normal_construction():
    with pytest.raises(DerivedFieldMismatchError) as exception_info:
        from_dict(
            Order,
            {
                "items": [{"amount": 2}, {"amount": 3}],
                "amounts": [2, 3],
                "total": 99,
            },
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert isinstance(exception_info.value, AssertionError)
    assert exception_info.value.field_path == "total"
    assert exception_info.value.expected_value == 99
    assert exception_info.value.actual_value == 5
    assert Item.post_init_count == 2
    assert Order.post_init_count == 1


def test_assert_policy_reports_nested_derived_field_mismatch():
    with pytest.raises(DerivedFieldMismatchError) as exception_info:
        from_dict(
            Order,
            {"items": [{"amount": 2, "total": 99}], "amounts": [2], "total": 2},
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert exception_info.value.field_path == "items.0.total"
    assert exception_info.value.expected_value == 99
    assert exception_info.value.actual_value == 2
    assert Item.post_init_count == 1
    assert Order.post_init_count == 0


def test_assert_policy_reports_field_inside_derived_nested_model_without_rebuilding_it():
    with pytest.raises(DerivedFieldMismatchError) as exception_info:
        from_dict(
            OrderWithSummary,
            {
                "items": [{"amount": 2}, {"amount": 3}],
                "summary": {"value": 99},
                "total": 5,
            },
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert exception_info.value.field_path == "summary.value"
    assert exception_info.value.expected_value == 99
    assert exception_info.value.actual_value == 5
    assert Item.post_init_count == 2
    assert OrderWithSummary.post_init_count == 1
    assert Summary.post_init_count == 1


def test_assert_policy_reports_field_inside_derived_collection_of_nested_models_without_rebuilding():
    with pytest.raises(DerivedFieldMismatchError) as exception_info:
        from_dict(
            OrderWithSummaries,
            {"items": [{"amount": 2}, {"amount": 3}], "summaries": [{"value": 2}, {"value": 99}]},
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert exception_info.value.field_path == "summaries.1.value"
    assert exception_info.value.expected_value == 99
    assert exception_info.value.actual_value == 3
    assert Item.post_init_count == 2
    assert OrderWithSummaries.post_init_count == 1
    assert Summary.post_init_count == 2


def test_assert_policy_applies_type_hooks_to_supplied_derived_value():
    result = from_dict(
        Order,
        {"items": [{"amount": 2}, {"amount": 3}], "amounts": [2, 3], "total": "5"},
        Config(
            derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH,
            type_hooks={int: lambda value: int(value)},
        ),
    )

    assert result.total == 5
    assert Order.post_init_count == 1


def test_assert_policy_distinguishes_type_errors_from_assertion_mismatches():
    with pytest.raises(WrongTypeError) as exception_info:
        from_dict(
            Order,
            {"items": [{"amount": 2}, {"amount": 3}], "amounts": [2, 3], "total": "5"},
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert exception_info.value.field_path == "total"
    assert not isinstance(exception_info.value, AssertionError)
    assert Item.post_init_count == 2
    assert Order.post_init_count == 1


def test_assert_policy_reports_collection_item_mismatch():
    with pytest.raises(DerivedFieldMismatchError) as exception_info:
        from_dict(
            Order,
            {"items": [{"amount": 2}, {"amount": 3}], "amounts": [2, 99], "total": 5},
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert exception_info.value.field_path == "amounts.1"
    assert exception_info.value.expected_value == 99
    assert exception_info.value.actual_value == 3


def test_reject_policy_rejects_supplied_derived_field():
    with pytest.raises(DerivedFieldInputError) as exception_info:
        from_dict(
            Order,
            {"items": [{"amount": 2}, {"amount": 3}], "total": 5},
            Config(derived_field_policy=DerivedFieldPolicy.REJECT_INPUT),
        )

    assert exception_info.value.field_path == "total"
    assert Order.post_init_count == 0


def test_reject_policy_reports_nested_derived_field():
    with pytest.raises(DerivedFieldInputError) as exception_info:
        from_dict(
            Order,
            {"items": [{"amount": 2, "total": 2}], "total": 2},
            Config(derived_field_policy=DerivedFieldPolicy.REJECT_INPUT),
        )

    assert exception_info.value.field_path == "items.0.total"
    assert Item.post_init_count == 0
    assert Order.post_init_count == 0


def test_reject_policy_reports_derived_nested_model_field():
    with pytest.raises(DerivedFieldInputError) as exception_info:
        from_dict(
            OrderWithSummary,
            {"items": [{"amount": 2}], "summary": {"value": 2}, "total": 2},
            Config(derived_field_policy=DerivedFieldPolicy.REJECT_INPUT),
        )

    assert exception_info.value.field_path == "summary"
    assert OrderWithSummary.post_init_count == 0


def test_derived_fields_are_optional_by_default():
    result = from_dict(Order, {"items": [{"amount": 2}, {"amount": 3}]})

    assert result.total == 5
    assert Item.post_init_count == 2
    assert Order.post_init_count == 1


def test_assert_policy_does_not_catch_dataclass_construction_errors():
    @dataclass
    class InvalidOrder:
        items: List[Item]
        total: int = field(init=False)

        def __post_init__(self) -> None:
            raise ValueError("invalid order")

    with pytest.raises(ValueError, match="invalid order"):
        from_dict(
            InvalidOrder,
            {"items": []},
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires Python 3.10")
def test_assert_policy_works_with_frozen_slots_dataclasses():
    @dataclass(frozen=True, slots=True)
    class FrozenOrder:
        items: List[int]
        total: int = field(init=False)
        post_init_count = 0

        def __post_init__(self) -> None:
            FrozenOrder.post_init_count += 1
            object.__setattr__(self, "total", sum(self.items))

    result = from_dict(
        FrozenOrder,
        {"items": [2, 3], "total": 5},
        Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
    )

    assert result.total == 5
    assert FrozenOrder.post_init_count == 1

    with pytest.raises(DerivedFieldMismatchError) as exception_info:
        from_dict(
            FrozenOrder,
            {"items": [2, 3], "total": 99},
            Config(derived_field_policy=DerivedFieldPolicy.ASSERT_MATCH),
        )

    assert exception_info.value.field_path == "total"
    assert FrozenOrder.post_init_count == 2
