# pylint: disable=unsupported-binary-operation
import sys
from dataclasses import Field, is_dataclass
from functools import reduce
from typing import Any, Dict, Generic, List, Set, Tuple, Type, TypeVar, Union, get_type_hints

from dacite.exceptions import DaciteError, TypeVarConstraintError

try:
    from typing import get_args, get_origin, Literal  # type: ignore
except ImportError:
    from typing_extensions import get_args, get_origin, Literal  # type: ignore


from .dataclasses import get_fields as dataclasses_get_fields


def __resolve_type_var(type_var: TypeVar, generics: Dict[TypeVar, Type]) -> Any:
    """Follows (type var, type) mappings until a non-type-var (or an unmapped type var) is found."""
    seen: Set[TypeVar] = set()
    while isinstance(type_var, TypeVar) and type_var in generics and type_var not in seen:
        seen.add(type_var)
        type_var = generics[type_var]  # type: ignore
    return type_var


def __resolve_forward_reference(hint: Any, data_class: Type) -> Any:
    """Resolves a hint (e.g. a type var bound) given as a string or a ForwardRef."""
    if isinstance(hint, str):
        return __dereference(hint, data_class)
    forward_arg = getattr(hint, "__forward_arg__", None)
    if forward_arg is not None:
        return __dereference(forward_arg, data_class)
    return hint


def __is_sub_type(sub_type: Any, base_type: Any) -> bool:
    sub_origin = get_origin(sub_type)
    if sub_origin is not None:
        sub_type = sub_origin
    base_origin = get_origin(base_type)
    if base_origin is not None:
        base_type = base_origin
    try:
        return issubclass(sub_type, base_type)
    except TypeError:
        return False


def __check_type_var_constraints(type_var: TypeVar, arg: Any, data_class: Type, path: List[str]) -> None:
    """Validates a concrete type against the constraints/bound of the type var it substitutes."""
    if isinstance(arg, TypeVar) or arg is Any:
        return
    constraints = getattr(type_var, "__constraints__", ())
    if constraints:
        for constraint in constraints:
            if __is_sub_type(arg, __resolve_forward_reference(constraint, data_class)):
                return
        raise TypeVarConstraintError(type_var=type_var, value=arg, path=tuple(path))
    bound = getattr(type_var, "__bound__", None)
    if bound is not None:
        if not __is_sub_type(arg, __resolve_forward_reference(bound, data_class)):
            raise TypeVarConstraintError(type_var=type_var, value=arg, path=tuple(path))


def __path_entry(hint: Any) -> str:
    origin = get_origin(hint)
    if origin is None:
        return getattr(hint, "__qualname__", str(hint))
    return str(hint)


def __add_generics(
    type_origin: Any, type_args: Tuple, generics: Dict[TypeVar, Type], data_class: Type, path: List[str]
) -> None:
    """Adds (type var, concrete type) entries derived from a type's origin and args to the provided generics dict."""
    if type_origin and type_args and hasattr(type_origin, "__parameters__"):
        for param, arg in zip(type_origin.__parameters__, type_args):
            if isinstance(param, TypeVar):
                arg = __resolve_type_var(arg, generics)
                if param in generics and generics[param] != arg:
                    raise DaciteError(f"Ambiguous TypeVar: {generics[param]} != {arg}")
                __check_type_var_constraints(param, arg, data_class, path)
                generics[param] = arg


def __collect_generics(
    hint: Any, generics: Dict[TypeVar, Type], data_class: Type, path: List[str], visited: Set[Any]
) -> None:
    """
    Recursively collects (type var, concrete type) mappings from a (possibly generic) class
    and its (generic) ancestors, validating each specialization against the declared
    constraints/bound of the substituted type var.
    """
    if hint in visited:
        return
    visited.add(hint)
    cls = get_origin(hint) or hint
    if not isinstance(cls, type):
        return
    path = path + [__path_entry(hint)]
    __add_generics(cls, get_args(hint), generics, data_class, path)
    bases = cls.__dict__.get("__orig_bases__", cls.__bases__)
    for base in bases:
        if base is object or base is Generic or get_origin(base) is Generic:
            continue
        __collect_generics(base, generics, data_class, path, visited)


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
        hint = __resolve_type_var(hint, generics)
        if isinstance(hint, TypeVar):
            # The type var was not specialized - fall back on the documented default:
            # its bound, the union of its constraints or Any (in this order of precedence).
            return __type_var_default(hint, generics, data_class)
        return hint

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


def __type_var_default(type_var: TypeVar, generics: Dict[TypeVar, Type], data_class: Type) -> Any:
    """Provides the documented default for a type var that was not specialized."""
    bound = getattr(type_var, "__bound__", None)
    if bound is not None:
        bound = __resolve_forward_reference(bound, data_class)
        if bound is type_var:
            return Any
        return __concretize(bound, generics, data_class)
    constraints = getattr(type_var, "__constraints__", ())
    if constraints:
        return Union[
            tuple(__concretize(__resolve_forward_reference(c, data_class), generics, data_class) for c in constraints)
        ]
    return Any


def orig(data_class: Type) -> Any:
    if is_dataclass(data_class):
        return data_class
    return get_origin(data_class)


def get_concrete_type_hints(data_class: Type, *args, **kwargs) -> Dict[str, Any]:
    """
    An overwrite of typing.get_type_hints supporting generics and forward references,
    i.e. substituting concrete types in type vars and references.
    """
    generics: Dict[TypeVar, Type] = {}

    __collect_generics(data_class, generics, orig(data_class), [], set())

    data_class = orig(data_class)
    hints = get_type_hints(data_class, *args, **kwargs)

    for key, hint in hints.copy().items():
        hints[key] = __concretize(hint, generics, data_class)

    return hints


def get_fields(data_class: Type) -> List[Field]:
    """An overwrite of dacite.dataclasses.get_fields supporting generics."""
    return dataclasses_get_fields(orig(data_class))
