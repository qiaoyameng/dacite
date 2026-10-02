# pylint: disable=unsupported-binary-operation
import sys
from dataclasses import Field, is_dataclass
from functools import reduce
from typing import Any, Dict, ForwardRef, Generic, List, Tuple, Type, TypeVar, get_type_hints

from dacite.exceptions import DaciteError, TypeVarBoundError

try:
    from typing import get_args, get_origin, Literal  # type: ignore
except ImportError:
    from typing_extensions import get_args, get_origin, Literal  # type: ignore


from .dataclasses import get_fields as dataclasses_get_fields


def __add_generics(
    type_origin: Any,
    type_args: Tuple,
    generics: Dict[TypeVar, Type],
    paths: Dict[TypeVar, Tuple[str, ...]],
    path: Tuple[str, ...],
) -> None:
    """Adds (type var, concrete type) entries derived from a type's origin and args to the provided generics dict."""
    if type_origin and type_args and hasattr(type_origin, "__parameters__"):
        for param, arg in zip(type_origin.__parameters__, type_args):
            if isinstance(param, TypeVar):
                if param in generics and generics[param] != arg:
                    raise DaciteError(f"Ambiguous TypeVar: {generics[param]} != {arg}")
                generics[param] = arg
                paths[param] = path


def __format_specialization(type_origin: Any, type_args: Tuple) -> str:
    name = getattr(type_origin, "__name__", str(type_origin))
    args = ", ".join(getattr(arg, "__name__", str(arg)) for arg in type_args)
    return f"{name}[{args}]"


def __resolve_type_var(hint: Any, generics: Dict[TypeVar, Type]) -> Any:
    """
    Transitively resolve a type var through the generics dict, e.g. when a concrete type
    is passed along multiple inheritance levels via distinct (same-named) type vars.
    """
    seen = set()
    while isinstance(hint, TypeVar) and hint in generics and hint not in seen:
        seen.add(hint)
        hint = generics[hint]
    return hint


def __dereference(type_name: str, data_class: Type) -> Type:
    """
    Try to find the class belonging to the reference in the provided module and,
    if not found, iteratively look in parent modules.
    """
    if data_class.__class__.__name__ == type_name:
        return data_class

    module_name = data_class.__module__
    parts = module_name.split(".")
    for i in range(len(parts)):
        try:
            module = sys.modules[".".join(parts[:-i]) if i else module_name]
            return getattr(module, type_name)
        except AttributeError:
            pass
    raise AttributeError("Could not find reference.")


def __concretize(hint: Any, generics: Dict[TypeVar, Type], data_class: Type) -> Any:
    """Recursively replace type vars and forward references by concrete types."""

    if isinstance(hint, str):
        return __dereference(hint, data_class)

    if isinstance(hint, TypeVar):
        # Fall back on the original TypeVar if the generics dict does not contain it.
        # Setting config.check_types=False will in some cases still make from_dict work, albeit not type checked ofc.
        return __resolve_type_var(hint, generics)

    hint_origin = get_origin(hint)
    hint_args = get_args(hint)
    if hint_origin and hint_args and hint_origin is not Literal:
        concrete_hint_args = tuple(__concretize(a, generics, data_class) for a in hint_args)
        if concrete_hint_args != hint_args:
            if sys.version_info >= (3, 9):
                if sys.version_info >= (3, 10) and hint_origin is type(int | str):
                    # avoid "UnionType is not subscriptable"
                    return reduce(lambda a, b: a | b, concrete_hint_args)
                # construct a copy with concretized args
                return hint_origin[concrete_hint_args]
            # It's generally not a good practice to overwrite __args__,
            # and it even has become impossible starting from python 3.13 (read-only),
            # but changing the output of get_type_hints is harmless (see unit test)
            # and at least this way, we get it working for python 3.8.
            hint.__args__ = concrete_hint_args

    return hint


def __collect_generics(data_class: Type) -> Tuple[Dict[TypeVar, Type], Dict[TypeVar, Tuple[str, ...]]]:
    """
    Collect (type var, concrete type) bindings for the given (possibly parameterized) data class,
    propagating concrete types through the whole inheritance hierarchy. Type vars are dict keys,
    so same-named but distinct type vars never collide. For each binding, the inheritance path
    leading to it is recorded for error reporting.
    """
    generics: Dict[TypeVar, Type] = {}
    paths: Dict[TypeVar, Tuple[str, ...]] = {}

    dc_origin = get_origin(data_class)
    dc_args = get_args(data_class)
    if dc_origin is not None and dc_args:
        __add_generics(dc_origin, dc_args, generics, paths, (__format_specialization(dc_origin, dc_args),))

    data_class = orig(data_class)
    mro = getattr(data_class, "__mro__", ())
    for index, klass in enumerate(mro):
        path = tuple(base.__name__ for base in mro[: index + 1])
        for base in klass.__dict__.get("__orig_bases__", ()):
            base_origin = get_origin(base)
            if base_origin is None or base_origin is Generic:
                continue
            concrete_args = tuple(__concretize(arg, generics, data_class) for arg in get_args(base))
            base_path = path + (__format_specialization(base_origin, concrete_args),)
            __add_generics(base_origin, concrete_args, generics, paths, base_path)

    return generics, paths


def __satisfies_bound(arg: Any, bound: Any, data_class: Type) -> bool:
    """Check whether a concrete type satisfies a type var's bound or one of its constraints."""
    if isinstance(bound, str):
        bound = __dereference(bound, data_class)
    if isinstance(bound, ForwardRef):
        bound = __dereference(bound.__forward_arg__, data_class)
    if bound is Any or isinstance(bound, TypeVar):
        return True
    if arg is Any:
        return True
    if isinstance(arg, ForwardRef):
        arg = __dereference(arg.__forward_arg__, data_class)
    if arg == bound:
        return True
    # As described in PEP 484 - section: "The numeric tower"
    if bound is float and arg is int:
        return True
    if bound is complex and arg in (int, float):
        return True
    arg_origin = get_origin(arg)
    if arg_origin is not None:
        arg = arg_origin
    bound_origin = get_origin(bound)
    if bound_origin is not None:
        bound = bound_origin
    try:
        return issubclass(arg, bound)
    except TypeError:
        return False


def __validate_type_var_bindings(
    generics: Dict[TypeVar, Type], paths: Dict[TypeVar, Tuple[str, ...]], data_class: Type
) -> None:
    """Validate every specialized type var against its declared bound or constraints."""
    for type_var, arg in generics.items():
        arg = __resolve_type_var(arg, generics)
        if isinstance(arg, TypeVar):
            # Unspecialized type var - falls back on the TypeVar itself (see __concretize).
            continue
        path = list(paths.get(type_var, ()))
        if type_var.__constraints__:
            if not any(__satisfies_bound(arg, constraint, data_class) for constraint in type_var.__constraints__):
                raise TypeVarBoundError(type_var=type_var, value=arg, path=path)
        elif type_var.__bound__ is not None:
            if not __satisfies_bound(arg, type_var.__bound__, data_class):
                raise TypeVarBoundError(type_var=type_var, value=arg, path=path)


def orig(data_class: Type) -> Any:
    if is_dataclass(data_class):
        return data_class
    return get_origin(data_class)


def get_concrete_type_hints(data_class: Type, *args, **kwargs) -> Dict[str, Any]:
    """
    An overwrite of typing.get_type_hints supporting generics and forward references,
    i.e. substituting concrete types in type vars and references.
    """
    generics, paths = __collect_generics(data_class)

    data_class = orig(data_class)

    __validate_type_var_bindings(generics, paths, data_class)

    hints = get_type_hints(data_class, *args, **kwargs)

    for key, hint in hints.copy().items():
        hints[key] = __concretize(hint, generics, data_class)

    return hints


def get_fields(data_class: Type) -> List[Field]:
    """An overwrite of dacite.dataclasses.get_fields supporting generics."""
    return dataclasses_get_fields(orig(data_class))
