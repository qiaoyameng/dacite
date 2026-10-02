import sys
from typing import Any, Dict, Type

_REQUIRED_ORIGINS = set()

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, get_args, get_origin, get_type_hints

    _REQUIRED_ORIGINS.update({Required, NotRequired})
    _include_extras_supported = True
else:
    try:
        from typing_extensions import (
            NotRequired as ExtendedNotRequired,
            Required as ExtendedRequired,
            get_args,
            get_origin,
            get_type_hints,
        )
    except ImportError:
        from typing import get_args, get_origin, get_type_hints

        _include_extras_supported = sys.version_info >= (3, 9)
    else:
        _REQUIRED_ORIGINS.update({ExtendedRequired, ExtendedNotRequired})
        _include_extras_supported = True

if sys.version_info >= (3, 10):
    from typing import NotRequired, Required

    _REQUIRED_ORIGINS.update({Required, NotRequired})

from dacite.cache import cache


def extract_typed_dict_field_type(type_: Type) -> Type:
    if get_origin(type_) in _REQUIRED_ORIGINS:
        return get_args(type_)[0]
    return type_


@cache
def get_typed_dict_hints(
    typed_dict: Type, localns: Any = None, globalns: Any = None, include_extras: bool = True
) -> Dict[str, Any]:
    kwargs = {"localns": localns, "globalns": globalns}
    if _include_extras_supported:
        kwargs["include_extras"] = include_extras
    return get_type_hints(typed_dict, **kwargs)
