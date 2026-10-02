from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict


class AnnotationStage(Enum):
    BEFORE_TYPE_CONSTRUCTION = "before_type_construction"
    AFTER_TYPE_CONSTRUCTION = "after_type_construction"


class AnnotationProcessor(ABC):
    stage: AnnotationStage

    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata

    @abstractmethod
    def process(self, value: Any) -> Any:
        ...


class BeforeTypeConstructionProcessor(AnnotationProcessor):
    stage = AnnotationStage.BEFORE_TYPE_CONSTRUCTION


class AfterTypeConstructionProcessor(AnnotationProcessor):
    stage = AnnotationStage.AFTER_TYPE_CONSTRUCTION


ProcessorFactory = Callable[[Any], AnnotationProcessor]


def get_annotation_processor(metadata: Any, processors: Dict[Any, ProcessorFactory]) -> AnnotationProcessor:
    processor_factory = processors.get(type(metadata))
    if processor_factory is None:
        for annotation_type, candidate in processors.items():
            if metadata == annotation_type or (
                isinstance(annotation_type, type) and isinstance(metadata, annotation_type)
            ):
                processor_factory = candidate
                break
    if processor_factory is None:
        raise KeyError(metadata)
    return processor_factory(metadata)
