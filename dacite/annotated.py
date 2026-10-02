from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Tuple

from dacite.exceptions import AnnotatedMetadataError, AnnotatedValidationError


class AnnotatedPhase(Enum):
    TRANSFORM = "transform"
    VALIDATE = "validate"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AnnotatedHandler:
    """A handler explicitly registered for a single ``Annotated`` metadata item.

    ``TRANSFORM`` handlers run on the raw input data before the base type is
    constructed, ``VALIDATE`` handlers run on the value after construction.
    """

    phase: AnnotatedPhase
    func: Callable[[Any], Any]

    @classmethod
    def transform(cls, func: Callable[[Any], Any]) -> "AnnotatedHandler":
        return cls(phase=AnnotatedPhase.TRANSFORM, func=func)

    @classmethod
    def validate(cls, func: Callable[[Any], Any]) -> "AnnotatedHandler":
        return cls(phase=AnnotatedPhase.VALIDATE, func=func)


def resolve_annotated_handlers(metadata: tuple, config: Any) -> List[Tuple[Any, AnnotatedHandler]]:
    """Map metadata items to registered handlers, keeping declaration order.

    Unknown metadata is kept (ignored) for backwards compatibility unless
    ``config.strict_annotated`` is enabled.
    """
    handlers = []
    for item in metadata:
        try:
            handler = config.annotated_handlers[item]
        except KeyError:
            if config.strict_annotated:
                raise AnnotatedMetadataError(metadata=item) from None
            continue
        handlers.append((item, handler))
    return handlers


def run_annotated_phase(handlers: List[Tuple[Any, AnnotatedHandler]], phase: AnnotatedPhase, value: Any) -> Any:
    """Run every handler of the given phase exactly once, in declaration order."""
    for item, handler in handlers:
        if handler.phase is not phase:
            continue
        try:
            value = handler.func(value)
        except AnnotatedValidationError:
            raise
        except ValueError as error:
            raise AnnotatedValidationError(metadata=item, phase=phase, value=value, message=str(error)) from None
    return value
