from dataclasses import InitVar, is_dataclass
from typing import (
    Type,
    Any,
    Optional,
    Union,
    Collection,
    TypeVar,
    Mapping,
    Tuple,
    cast as typing_cast,
    get_type_hints,
)

try:
    from typing import get_origin, get_args  # type: ignore
except ImportError:
    from typing_extensions import get_origin, get_args  # type: ignore

from dacite.cache import cache

T = TypeVar("T", bound=Any)


@cache
def extract_origin_collection(collection: Type) -> Type:
    try:
        return collection.__extra__
    except AttributeError:
        return collection.__origin__


@cache
def is_optional(type_: Type) -> bool:
    return is_union(type_) and type(None) in extract_generic(type_)


@cache
def extract_optional(optional: Type[Optional[T]]) -> T:
    other_members = [member for member in extract_generic(optional) if member is not type(None)]
    if other_members:
        return typing_cast(T, Union[tuple(other_members)])
    else:
        raise ValueError("can not find not-none value")


@cache
def is_generic(type_: Type) -> bool:
    return hasattr(type_, "__origin__")


@cache
def is_union(type_: Type) -> bool:
    if is_generic(type_) and type_.__origin__ == Union:
        return True

    try:
        from types import UnionType  # type: ignore

        return isinstance(type_, UnionType)
    except ImportError:
        return False


@cache
def is_tuple(type_: Type) -> bool:
    return is_subclass(type_, tuple)


@cache
def is_literal(type_: Type) -> bool:
    try:
        from typing import Literal  # type: ignore

        return is_generic(type_) and type_.__origin__ == Literal
    except ImportError:
        return False


@cache
def is_new_type(type_: Type) -> bool:
    return hasattr(type_, "__supertype__")


@cache
def extract_new_type(type_: Type) -> Type:
    return type_.__supertype__


@cache
def is_init_var(type_: Type) -> bool:
    return isinstance(type_, InitVar) or type_ is InitVar


@cache
def is_typed_dict(type_: Type) -> bool:
    try:
        from typing import is_typeddict

        return is_typeddict(type_)
    except ImportError:
        try:
            return (
                issubclass(type_, dict)
                and hasattr(type_, "__required_keys__")
                and hasattr(type_, "__optional_keys__")
            )
        except TypeError:
            return False


@cache
def is_required_marker(type_: Type) -> bool:
    origin = get_origin(type_) if is_generic(type_) else None
    return getattr(origin, "__name__", "") == "Required" or getattr(origin, "_name", "") == "Required"


@cache
def is_not_required_marker(type_: Type) -> bool:
    origin = get_origin(type_) if is_generic(type_) else None
    return getattr(origin, "__name__", "") == "NotRequired" or getattr(origin, "_name", "") == "NotRequired"


@cache
def is_requiredness_marker(type_: Type) -> bool:
    return is_required_marker(type_) or is_not_required_marker(type_)


@cache
def get_typed_dict_hints(type_: Type, localns: Optional[Mapping[str, Any]] = None) -> dict:
    try:
        return get_type_hints(type_, localns=localns, include_extras=True)
    except TypeError:
        try:
            from typing_extensions import get_type_hints as get_extended_type_hints

            return get_extended_type_hints(type_, localns=localns, include_extras=True)
        except ImportError:
            return get_type_hints(type_, localns=localns)


@cache
def extract_init_var(type_: Type) -> Union[Type, Any]:
    try:
        return type_.type
    except AttributeError:
        return Any


@cache
def is_generic_collection(type_: Type) -> bool:
    if not is_generic(type_):
        return False
    origin = extract_origin_collection(type_)
    try:
        return bool(origin and issubclass(origin, Collection))
    except (TypeError, AttributeError):
        return False


@cache
def extract_generic(type_: Type, defaults: Tuple = ()) -> tuple:
    try:
        if getattr(type_, "_special", False):
            return defaults
        if type_.__args__ == ():
            return (type_.__args__,)
        return type_.__args__ or defaults  # type: ignore
    except AttributeError:
        return defaults


@cache
def is_subclass(sub_type: Type, base_type: Type) -> bool:
    if is_generic_collection(sub_type):
        sub_type = extract_origin_collection(sub_type)
    try:
        return issubclass(sub_type, base_type)
    except TypeError:
        return False


@cache
def is_type_generic(type_: Type) -> bool:
    try:
        return type_.__origin__ in (type, Type)
    except AttributeError:
        return False


@cache
def is_generic_dataclass(type_: Type) -> bool:
    return is_dataclass(get_origin(type_))


def is_instance(value: Any, type_: Type, localns: Optional[Mapping[str, Any]] = None) -> bool:
    try:
        # As described in PEP 484 - section: "The numeric tower"
        if (type_ in [float, complex] and isinstance(value, (int, float))) or isinstance(value, type_):
            return True
    except TypeError:
        pass
    if type_ == Any:
        return True
    if is_requiredness_marker(type_):
        type_ = get_args(type_)[0]
    if is_union(type_):
        return any(is_instance(value, t, localns=localns) for t in extract_generic(type_))
    if is_typed_dict(type_):
        if not isinstance(value, Mapping):
            return False
        try:
            hints = get_typed_dict_hints(type_, localns=localns)
        except Exception:  # pylint: disable=broad-except
            return False
        if any(key not in value for key in type_.__required_keys__):
            return False
        for key, item in value.items():
            if key in hints:
                item_type = hints[key]
                if is_requiredness_marker(item_type):
                    item_type = get_args(item_type)[0]
                if not is_instance(item, item_type, localns=localns):
                    return False
        return True
    if is_generic_collection(type_):
        origin = extract_origin_collection(type_)
        if not isinstance(value, origin):
            return False
        if not extract_generic(type_):
            return True
        if isinstance(value, tuple) and is_tuple(type_):
            tuple_types = extract_generic(type_)
            if len(tuple_types) == 1 and tuple_types[0] == ():
                return len(value) == 0
            if len(tuple_types) == 2 and tuple_types[1] is ...:
                return all(is_instance(item, tuple_types[0], localns=localns) for item in value)
            if len(tuple_types) != len(value):
                return False
            return all(is_instance(item, item_type, localns=localns) for item, item_type in zip(value, tuple_types))
        if isinstance(value, Mapping):
            key_type, val_type = extract_generic(type_, defaults=(Any, Any))
            for key, val in value.items():
                if not is_instance(key, key_type, localns=localns) or not is_instance(
                    val, val_type, localns=localns
                ):
                    return False
            return True
        return all(
            is_instance(item, extract_generic(type_, defaults=(Any,))[0], localns=localns) for item in value
        )
    if is_new_type(type_):
        return is_instance(value, extract_new_type(type_), localns=localns)
    if is_literal(type_):
        return value in extract_generic(type_)
    if is_init_var(type_):
        return is_instance(value, extract_init_var(type_), localns=localns)
    if is_type_generic(type_):
        return is_subclass(value, extract_generic(type_)[0])
    if is_generic_dataclass(type_):
        return isinstance(value, get_origin(type_))  # type: ignore[arg-type]
    return False
