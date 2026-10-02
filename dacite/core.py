from dataclasses import is_dataclass
from itertools import repeat, zip_longest
from typing import TypeVar, Type, Optional, Mapping, Any, Collection, MutableMapping

from dacite.cache import cache
from dacite.config import Config, DerivedFieldPolicy
from dacite.data import Data
from dacite.dataclasses import (
    get_default_value_for_field,
    DefaultValueNotFoundError,
    is_frozen,
    is_init_var_field,
)
from dacite.exceptions import (
    ForwardReferenceError,
    WrongTypeError,
    DaciteError,
    UnionMatchError,
    MissingValueError,
    DaciteFieldError,
    UnexpectedDataError,
    StrictUnionMatchError,
    DerivedFieldError,
    DerivedFieldInputError,
    DerivedFieldMismatchError,
)
from dacite.types import (
    is_instance,
    is_generic_collection,
    is_union,
    extract_generic,
    is_optional,
    extract_origin_collection,
    is_init_var,
    extract_init_var,
    is_subclass,
)

from dacite.generics import get_concrete_type_hints, get_fields, orig

T = TypeVar("T")


def from_dict(data_class: Type[T], data: Data, config: Optional[Config] = None) -> T:
    """Create a data class instance from a dictionary.

    :param data_class: a data class type
    :param data: a dictionary of a input data
    :param config: a configuration of the creation process
    :return: an instance of a data class
    """
    init_values: MutableMapping[str, Any] = {}
    post_init_values: MutableMapping[str, Any] = {}
    derived_assertion_values: MutableMapping[str, Any] = {}
    config = config or Config()

    try:
        data_class_hints = cache(get_concrete_type_hints)(data_class, localns=config.hashable_forward_references)
    except NameError as error:
        raise ForwardReferenceError(str(error)) from None
    data_class_fields = cache(get_fields)(data_class)

    if config.strict:
        extra_fields = set(data.keys()) - {f.name for f in data_class_fields}
        if extra_fields:
            raise UnexpectedDataError(keys=extra_fields)
    for field in data_class_fields:
        field_type = data_class_hints[field.name]
        key = config.convert_key(field.name)

        if key in data:
            if not field.init:
                if config.derived_field_policy == DerivedFieldPolicy.REJECT_INPUT:
                    raise DerivedFieldInputError(field_path=field.name)
                if config.derived_field_policy == DerivedFieldPolicy.ASSERT_MATCH:
                    derived_assertion_values[field.name] = data[key]
                    continue

            try:
                value = _build_value(type_=field_type, data=data[key], config=config)
            except DaciteFieldError as error:
                error.update_path(field.name)
                raise
            if config.check_types and not is_instance(value, field_type):
                raise WrongTypeError(field_path=field.name, field_type=field_type, value=value)
        else:
            try:
                value = get_default_value_for_field(field, field_type)
            except DefaultValueNotFoundError:
                if not field.init:
                    continue
                raise MissingValueError(field.name) from None
        if field.init:
            init_values[field.name] = value
        elif not is_frozen(data_class):
            post_init_values[field.name] = value

    instance = data_class(**init_values)

    for key, value in post_init_values.items():
        setattr(instance, key, value)

    _assert_derived_fields(
        instance=instance,
        data=data,
        hints=data_class_hints,
        config=config,
        derived_fields=derived_assertion_values,
    )

    return instance


def _build_value(type_: Type, data: Any, config: Config) -> Any:
    if type_ in config.type_hooks:
        data = config.type_hooks[type_](data)
    return _build_value_without_type_hook(type_=type_, data=data, config=config)


def _build_value_without_type_hook(type_: Type, data: Any, config: Config) -> Any:
    if is_init_var(type_):
        type_ = extract_init_var(type_)
    if is_optional(type_) and data is None:
        return data
    if is_union(type_):
        data = _build_value_for_union(union=type_, data=data, config=config)
    elif is_generic_collection(type_):
        data = _build_value_for_collection(collection=type_, data=data, config=config)
    elif cache(is_dataclass)(orig(type_)) and isinstance(data, Mapping):
        data = from_dict(data_class=type_, data=data, config=config)
    for cast_type in config.cast:
        if is_subclass(type_, cast_type):
            if is_generic_collection(type_):
                data = extract_origin_collection(type_)(data)
            else:
                data = type_(data)
            break
    return data


def _assert_derived_fields(
    instance: Any,
    data: Mapping[str, Any],
    hints: Mapping[str, Type[Any]],
    config: Config,
    derived_fields: Mapping[str, Any],
) -> None:
    for field_name in derived_fields:
        key = config.convert_key(field_name)
        actual_value = getattr(instance, field_name)
        _assert_value(
            type_=hints[field_name],
            expected_value=data[key],
            actual_value=actual_value,
            config=config,
            field_path=field_name,
        )


def _assert_value(
    type_: Type[Any],
    expected_value: Any,
    actual_value: Any,
    config: Config,
    field_path: str,
    root_path: Optional[str] = None,
    apply_type_hook: bool = True,
) -> None:
    if apply_type_hook and type_ in config.type_hooks:
        expected_value = config.type_hooks[type_](expected_value)

    if is_init_var(type_):
        type_ = extract_init_var(type_)

    if is_optional(type_) and expected_value is None:
        if actual_value is not None:
            raise DerivedFieldMismatchError(field_path, expected_value, actual_value)
        return

    if is_union(type_):
        _assert_union_value(type_, expected_value, actual_value, config, field_path)
    elif is_generic_collection(type_):
        if any(is_subclass(type_, cast_type) for cast_type in config.cast):
            expected_value = extract_origin_collection(type_)(expected_value)
        _assert_collection_value(type_, expected_value, actual_value, config, field_path)
    elif cache(is_dataclass)(orig(type_)) and isinstance(expected_value, Mapping):
        _assert_dataclass_value(type_, expected_value, actual_value, config, root_path or field_path)
    else:
        expected_value = _build_value_without_type_hook(type_, expected_value, config)
        if config.check_types and not is_instance(expected_value, type_):
            raise WrongTypeError(field_path=field_path, field_type=type_, value=expected_value)
        if expected_value != actual_value:
            raise DerivedFieldMismatchError(field_path, expected_value, actual_value)


def _assert_union_value(
    union: Type[Any],
    expected_value: Any,
    actual_value: Any,
    config: Config,
    field_path: str,
) -> None:
    for inner_type in extract_generic(union):
        try:
            _assert_value(inner_type, expected_value, actual_value, config, field_path, apply_type_hook=False)
            return
        except DaciteError:
            continue
    if config.check_types:
        raise UnionMatchError(field_type=union, value=expected_value, field_path=field_path)
    if expected_value != actual_value:
        raise DerivedFieldMismatchError(field_path, expected_value, actual_value)


def _assert_collection_value(
    collection: Type[Any],
    expected_value: Any,
    actual_value: Any,
    config: Config,
    field_path: str,
) -> None:
    origin_collection = extract_origin_collection(collection)
    if not isinstance(expected_value, origin_collection) or not isinstance(actual_value, origin_collection):
        if config.check_types:
            raise WrongTypeError(field_path=field_path, field_type=collection, value=expected_value)
        if expected_value != actual_value:
            raise DerivedFieldMismatchError(field_path, expected_value, actual_value)
        return

    if len(expected_value) != len(actual_value):
        raise DerivedFieldMismatchError(field_path, expected_value, actual_value)

    if isinstance(expected_value, Mapping) and is_subclass(collection, Mapping):
        item_type = extract_generic(collection, defaults=(Any, Any))[1]
        if set(expected_value.keys()) != set(actual_value.keys()):
            raise DerivedFieldMismatchError(field_path, expected_value, actual_value)
        for key, value in expected_value.items():
            _assert_value(
                item_type,
                value,
                actual_value[key],
                config,
                f"{field_path}.{key}",
                root_path=f"{field_path}.{key}",
            )
        return

    if is_subclass(collection, tuple):
        types = extract_generic(collection)
        values = (
            zip_longest(expected_value, repeat(types[0]))
            if len(types) == 2 and types[1] is Ellipsis
            else zip_longest(expected_value, types)
        )
        for index, (value, item_type) in enumerate(values):
            _assert_value(
                item_type,
                value,
                actual_value[index],
                config,
                f"{field_path}.{index}",
                root_path=f"{field_path}.{index}",
            )
        return

    if isinstance(expected_value, (set, frozenset)) and isinstance(actual_value, (set, frozenset)):
        remaining_actual_values = list(actual_value)
        for expected_item in expected_value:
            for actual_item in remaining_actual_values:
                try:
                    _assert_value(item_type, expected_item, actual_item, config, field_path, root_path=field_path)
                except DaciteError:
                    continue
                remaining_actual_values.remove(actual_item)
                break
            else:
                raise DerivedFieldMismatchError(field_path, expected_value, actual_value)
        return

    if is_subclass(collection, Collection):
        item_type = extract_generic(collection, defaults=(Any,))[0]
        for index, (expected_item, actual_item) in enumerate(zip_longest(expected_value, actual_value)):
            _assert_value(
                item_type,
                expected_item,
                actual_item,
                config,
                f"{field_path}.{index}",
                root_path=f"{field_path}.{index}",
            )


def _assert_dataclass_value(
    data_class: Type[Any],
    expected_value: Mapping[str, Any],
    actual_value: Any,
    config: Config,
    field_path: str,
) -> None:
    if config.strict:
        extra_fields = set(expected_value.keys()) - {field.name for field in cache(get_fields)(data_class)}
        if extra_fields:
            raise UnexpectedDataError(keys=extra_fields)

    if not isinstance(actual_value, orig(data_class)):
        if config.check_types:
            raise WrongTypeError(field_path=field_path, field_type=data_class, value=expected_value)
        if expected_value != actual_value:
            raise DerivedFieldMismatchError(field_path, expected_value, actual_value)
        return

    try:
        hints = cache(get_concrete_type_hints)(data_class, localns=config.hashable_forward_references)
    except NameError as error:
        raise ForwardReferenceError(str(error)) from None

    for field in cache(get_fields)(data_class):
        if is_init_var_field(field):
            continue
        key = config.convert_key(field.name)
        if key not in expected_value:
            if not field.init:
                continue
            try:
                expected_field_value = get_default_value_for_field(field, hints[field.name])
            except DefaultValueNotFoundError:
                raise MissingValueError(field_path=f"{field_path}.{field.name}") from None
        else:
            expected_field_value = expected_value[key]

        if not field.init:
            if config.derived_field_policy == DerivedFieldPolicy.REJECT_INPUT:
                raise DerivedFieldInputError(field_path=f"{field_path}.{field.name}")
            if config.derived_field_policy != DerivedFieldPolicy.ASSERT_MATCH:
                continue

        _assert_value(
            type_=hints[field.name],
            expected_value=expected_field_value,
            actual_value=getattr(actual_value, field.name),
            config=config,
            field_path=f"{field_path}.{field.name}",
        )


def _build_value_for_union(union: Type, data: Any, config: Config) -> Any:
    types = extract_generic(union)
    if is_optional(union) and len(types) == 2:
        return _build_value(type_=types[0], data=data, config=config)
    union_matches = {}
    for inner_type in types:
        try:
            # noinspection PyBroadException
            try:
                value = _build_value(type_=inner_type, data=data, config=config)
            except Exception:  # pylint: disable=broad-except
                continue
            if is_instance(value, inner_type):
                if config.strict_unions_match:
                    union_matches[inner_type] = value
                else:
                    return value
        except DaciteError:
            pass
    if config.strict_unions_match and union_matches:
        if len(union_matches) > 1:
            raise StrictUnionMatchError(union_matches)
        return union_matches.popitem()[1]
    if not config.check_types:
        return data
    raise UnionMatchError(field_type=union, value=data)


def _build_value_for_collection(collection: Type, data: Any, config: Config) -> Any:
    data_type = data.__class__
    if isinstance(data, Mapping) and is_subclass(collection, Mapping):
        item_type = extract_generic(collection, defaults=(Any, Any))[1]
        def build_mapping_value(key: Any, value: Any) -> Any:
            try:
                return _build_value(type_=item_type, data=value, config=config)
            except DerivedFieldError as error:
                error.update_path(str(key))
                raise

        return data_type((key, build_mapping_value(key, value)) for key, value in data.items())
    elif is_subclass(collection, tuple):
        if not data:
            return data_type()
        types = extract_generic(collection)

        def build_collection_item(index: int, item: Any, type_: Type) -> Any:
            try:
                return _build_value(type_=type_, data=item, config=config)
            except DerivedFieldError as error:
                error.update_path(str(index))
                raise

        if len(types) == 2 and types[1] == Ellipsis:
            return data_type(build_collection_item(index, item, types[0]) for index, item in enumerate(data))
        return data_type(
            build_collection_item(index, item, type_)
            for index, (item, type_) in enumerate(zip_longest(data, types))
        )
    elif isinstance(data, Collection) and is_subclass(collection, Collection):
        item_type = extract_generic(collection, defaults=(Any,))[0]

        def build_collection_item(index: int, item: Any) -> Any:
            try:
                return _build_value(type_=item_type, data=item, config=config)
            except DerivedFieldError as error:
                error.update_path(str(index))
                raise

        return data_type(
            build_collection_item(index, item) for index, item in enumerate(data)
        )
    return data
