from dataclasses import is_dataclass
from itertools import zip_longest
from typing import TypeVar, Type, Optional, Mapping, Any, Collection, MutableMapping

from dacite.cache import cache
from dacite.annotations import (
    AnnotationStage,
    get_annotation_processor,
)
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
    AnnotationError,
    AnnotationProcessingError,
    UnknownAnnotationError,
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
    is_annotated,
    extract_annotated_type,
    strip_annotations,
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
            try:
                value = _build_value(type_=field_type, data=data[key], config=config)
            except DaciteFieldError as error:
                error.update_path(field.name)
                raise
            if config.check_types and not is_instance(value, field_type):
                raise WrongTypeError(field_path=field.name, field_type=field_type, value=value)
        else:
            try:
                value = get_default_value_for_field(field, strip_annotations(field_type))
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
    if is_annotated(type_):
        return _build_annotated_value(type_=type_, data=data, config=config)
    return _build_unannotated_value(type_=type_, data=data, config=config)


def _build_annotated_value(type_: Type, data: Any, config: Config) -> Any:
    underlying_type = strip_annotations(type_)
    if is_optional(underlying_type) and data is None:
        return data

    metadata = type_.__metadata__
    processors = []
    for item in metadata:
        try:
            processor = get_annotation_processor(item, config.annotation_processors)
        except KeyError:
            if config.strict:
                raise UnknownAnnotationError(
                    metadata=item,
                    stage=AnnotationStage.BEFORE_TYPE_CONSTRUCTION,
                ) from None
            continue
        processors.append(processor)

    for processor in processors:
        if processor.stage == AnnotationStage.BEFORE_TYPE_CONSTRUCTION:
            try:
                data = processor.process(data)
            except Exception as error:  # pylint: disable=broad-except
                raise AnnotationProcessingError(
                    metadata=processor.metadata,
                    stage=processor.stage,
                    value=data,
                    original_error=error,
                ) from error

    data = _build_value(type_=extract_annotated_type(type_), data=data, config=config)

    for processor in processors:
        if processor.stage == AnnotationStage.AFTER_TYPE_CONSTRUCTION:
            try:
                data = processor.process(data)
            except Exception as error:  # pylint: disable=broad-except
                raise AnnotationProcessingError(
                    metadata=processor.metadata,
                    stage=processor.stage,
                    value=data,
                    original_error=error,
                ) from error
    return data


def _build_unannotated_value(type_: Type, data: Any, config: Config) -> Any:
    unannotated_type = strip_annotations(type_)
    type_hook = _get_type_hook(type_, config)
    if type_hook is not None:
        data = type_hook(data)
    if is_optional(unannotated_type) and data is None:
        return data
    if is_union(unannotated_type):
        data = _build_value_for_union(union=unannotated_type, data=data, config=config)
    elif is_generic_collection(unannotated_type):
        data = _build_value_for_collection(collection=unannotated_type, data=data, config=config)
    elif cache(is_dataclass)(orig(unannotated_type)) and isinstance(data, Mapping):
        data = from_dict(data_class=unannotated_type, data=data, config=config)
    for cast_type in config.cast:
        if is_subclass(unannotated_type, cast_type):
            if is_generic_collection(unannotated_type):
                data = extract_origin_collection(unannotated_type)(data)
            else:
                data = unannotated_type(data)
            break
    return data


def _get_type_hook(type_: Type, config: Config):
    if type_ in config.type_hooks:
        return config.type_hooks[type_]
    unannotated_type = strip_annotations(type_)
    if unannotated_type in config.type_hooks:
        return config.type_hooks[unannotated_type]
    if is_generic_collection(unannotated_type):
        origin = extract_origin_collection(unannotated_type)
        if origin in config.type_hooks:
            return config.type_hooks[origin]
        for hook_type, hook in config.type_hooks.items():
            hook_type = strip_annotations(hook_type)
            if (
                not is_generic_collection(hook_type)
                or extract_origin_collection(hook_type) != origin
            ):
                continue
            field_args = extract_generic(unannotated_type, defaults=(Any,))
            hook_args = extract_generic(hook_type, defaults=(Any,))
            stripped_field_args = tuple(strip_annotations(arg) for arg in field_args)
            if field_args == hook_args or stripped_field_args == hook_args:
                return hook
    return None


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
        return data_type(
            (key, _build_collection_value(item_type=item_type, data=value, config=config, path=str(key)))
            for key, value in data.items()
        )
    elif is_subclass(collection, tuple):
        if not data:
            return data_type()
        types = extract_generic(collection)
        if len(types) == 2 and types[1] == Ellipsis:
            return data_type(
                _build_collection_value(item_type=types[0], data=item, config=config, path=str(index))
                for index, item in enumerate(data)
            )
        return data_type(
            _build_collection_value(item_type=type_, data=item, config=config, path=str(index))
            for index, (item, type_) in enumerate(zip_longest(data, types))
        )
    elif isinstance(data, Collection) and is_subclass(collection, Collection):
        item_type = extract_generic(collection, defaults=(Any,))[0]
        return data_type(
            _build_collection_value(item_type=item_type, data=item, config=config, path=str(index))
            for index, item in enumerate(data)
        )
    return data


def _build_collection_value(item_type: Type, data: Any, config: Config, path: str) -> Any:
    try:
        return _build_value(type_=item_type, data=data, config=config)
    except AnnotationError as error:
        error.update_path(path)
        raise
