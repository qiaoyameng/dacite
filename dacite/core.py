from dataclasses import is_dataclass
from itertools import zip_longest
from typing import TypeVar, Type, Optional, Mapping, Any, Collection, MutableMapping

from dacite.cache import cache
from dacite.config import Config
from dacite.data import Data
from dacite.dataclasses import (
    get_default_value_for_field,
    DefaultValueNotFoundError,
    is_frozen,
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
    is_typed_dict,
)

from dacite.generics import get_concrete_type_hints, get_fields, orig
from dacite.typed_dict import (
    extract_typed_dict_field_type,
    get_typed_dict_hints,
)

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
    config = config or Config()

    try:
        data_class_hints = cache(get_concrete_type_hints)(data_class, localns=config.hashable_forward_references)
    except NameError as error:
        raise ForwardReferenceError(str(error)) from None
    data_class_fields = cache(get_fields)(data_class)

    if config.strict:
        expected_fields = {config.convert_key(f.name) for f in data_class_fields}
        extra_fields = set(data.keys()) - expected_fields
        if extra_fields:
            raise UnexpectedDataError(keys=extra_fields)

    for field in data_class_fields:
        field_type = data_class_hints[field.name]
        key = config.convert_key(field.name)

        if key in data:
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

    return instance


def _build_value(type_: Type, data: Any, config: Config) -> Any:
    if is_init_var(type_):
        type_ = extract_init_var(type_)
    if type_ in config.type_hooks:
        data = config.type_hooks[type_](data)
    if is_optional(type_) and data is None:
        return data
    if is_union(type_):
        data = _build_value_for_union(union=type_, data=data, config=config)
    elif is_generic_collection(type_):
        data = _build_value_for_collection(collection=type_, data=data, config=config)
    elif is_typed_dict(type_) and isinstance(data, Mapping):
        data = _build_typed_dict(typed_dict=type_, data=data, config=config)
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


def _build_typed_dict(typed_dict: Type, data: Mapping, config: Config) -> dict:
    try:
        hints = cache(get_typed_dict_hints)(typed_dict, localns=config.hashable_forward_references)
    except NameError as error:
        raise ForwardReferenceError(str(error)) from None

    if config.strict:
        expected_fields = {config.convert_key(field_name) for field_name in hints}
        extra_fields = set(data.keys()) - expected_fields
        if extra_fields:
            raise UnexpectedDataError(keys=extra_fields)

    result = {}
    required_fields = typed_dict.__required_keys__
    for field_name, field_type in hints.items():
        field_type = extract_typed_dict_field_type(field_type)
        key = config.convert_key(field_name)

        if key in data:
            try:
                value = _build_value(type_=field_type, data=data[key], config=config)
            except DaciteFieldError as error:
                error.update_path(field_name)
                raise
            if config.check_types and not is_instance(value, field_type):
                raise WrongTypeError(field_path=field_name, field_type=field_type, value=value)
            result[field_name] = value
        elif field_name in required_fields:
            raise MissingValueError(field_name)

    return result


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

        def build_mapping_value(key, value):
            try:
                return _build_value(type_=item_type, data=value, config=config)
            except DaciteFieldError as error:
                error.update_path("", index=key)
                raise

        return data_type((key, build_mapping_value(key, value)) for key, value in data.items())
    elif is_subclass(collection, tuple):
        if not data:
            return data_type()
        types = extract_generic(collection)
        if len(types) == 2 and types[1] == Ellipsis:
            def build_tuple_item(indexed_item):
                index, item = indexed_item
                try:
                    return _build_value(type_=types[0], data=item, config=config)
                except DaciteFieldError as error:
                    error.update_path("", index=index)
                    raise

            return data_type(build_tuple_item(item) for item in enumerate(data))

        def build_fixed_tuple_item(indexed_item, item_type):
            index, item = indexed_item
            try:
                return _build_value(type_=item_type, data=item, config=config)
            except DaciteFieldError as error:
                error.update_path("", index=index)
                raise

        return data_type(
            build_fixed_tuple_item(item, type_) for item, type_ in zip_longest(enumerate(data), types)
        )
    elif isinstance(data, Collection) and is_subclass(collection, Collection):
        item_type = extract_generic(collection, defaults=(Any,))[0]

        def build_collection_item(indexed_item):
            index, item = indexed_item
            try:
                return _build_value(type_=item_type, data=item, config=config)
            except DaciteFieldError as error:
                error.update_path("", index=index)
                raise

        return data_type(build_collection_item(item) for item in enumerate(data))
    return data
