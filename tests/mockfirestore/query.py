import warnings
from itertools import islice, tee
from typing import Any, AsyncGenerator, Callable, Generic, Iterator, List, Optional, TypeVar, Union

from mockfirestore._helpers import T
from mockfirestore.document import AsyncDocumentSnapshot, DocumentSnapshot


class Query:
    def __init__(
        self,
        parent: 'CollectionReference',
        projection=None,
        field_filters=(),
        orders=(),
        limit=None,
        offset=None,
        start_at=None,
        end_at=None,
        all_descendants=False,
    ) -> None:
        self.parent = parent
        self.projection = projection
        self._field_filters = []
        self.orders = list(orders)
        self._limit = limit
        self._offset = offset
        self._start_at = start_at
        self._end_at = end_at
        self.all_descendants = all_descendants

        if field_filters:
            for field_filter in field_filters:
                # Support both tuple filters and composite callable filters (Or/And)
                if callable(field_filter):
                    self._field_filters.append(field_filter)
                else:
                    self._add_field_filter(*field_filter)

    def stream(self, transaction=None) -> Iterator[DocumentSnapshot]:
        doc_snapshots = []
        for doc_snapshot in self.parent.stream():
            doc_snapshots.append(doc_snapshot)

        for filter in self._field_filters:
            if isinstance(filter, tuple):
                field, op, value = filter
                compare = self._compare_func(op)
                doc_snapshots = [
                    doc_snapshot
                    for doc_snapshot in doc_snapshots
                    if compare(doc_snapshot._get_by_field_path(field), value)
                ]
            else:  # Composite filter (And/Or)
                doc_snapshots = filter(doc_snapshots)

        if self.orders:
            for key, direction in self.orders:
                doc_snapshots = sorted(
                    doc_snapshots, key=lambda doc: doc.to_dict().get(key, None), reverse=direction == 'DESCENDING'
                )

        if self._start_at:
            document_fields_or_snapshot, before = self._start_at
            doc_snapshots = self._apply_cursor(document_fields_or_snapshot, doc_snapshots, before, True)

        if self._end_at:
            document_fields_or_snapshot, before = self._end_at
            doc_snapshots = self._apply_cursor(document_fields_or_snapshot, doc_snapshots, before, False)

        if self._offset:
            doc_snapshots = islice(doc_snapshots, self._offset, None)

        if self._limit:
            doc_snapshots = islice(doc_snapshots, self._limit)

        # Yield the documents to make this an async generator
        for doc_snapshot in doc_snapshots:
            yield doc_snapshot

    def get(self) -> Iterator[DocumentSnapshot]:
        warnings.warn(
            'Query.get is deprecated, please use Query.stream',
            category=DeprecationWarning,
        )
        return self.stream()

    def _add_field_filter(self, field: str, op: str, value: Any):
        if callable(field):  # This is a composite filter (And/Or)
            self._field_filters.append(field)
        else:
            self._field_filters.append((field, op, value))

    @staticmethod
    def make_field_filter(field: Optional[str], op: Optional[str], value: Any = None, filter=None):
        if bool(filter) and (bool(field) or bool(op)):
            raise ValueError("Can't pass in both the positional arguments and 'filter' at the same time")
        if filter:
            classname = filter.__class__.__name__
            if classname == 'Or':
                return lambda docs: [doc for doc in docs if any(Query._evaluate_filter(doc, f) for f in filter.filters)]
            elif classname == 'And':
                return lambda docs: [doc for doc in docs if all(Query._evaluate_filter(doc, f) for f in filter.filters)]
            elif classname.endswith('FieldFilter'):
                return (filter.field_path, filter.op_string, filter.value)
            else:
                raise NotImplementedError('Unsupported filter type: %s' % classname)
        else:
            return (field, op, value)

    @staticmethod
    def _evaluate_filter(doc: DocumentSnapshot, filter) -> bool:
        if not hasattr(filter, 'field_path'):
            return False

        field_value = doc._get_by_field_path(filter.field_path)
        if field_value is None:
            return False

        compare_func = Query._compare_func(filter.op_string)
        try:
            return compare_func(field_value, filter.value)
        except (TypeError, ValueError):
            return False

    def where(
        self,
        field: Optional[str] = None,
        op: Optional[str] = None,
        value: Any = None,
        filter=None,
    ) -> 'Query':
        filter_result = self.make_field_filter(field, op, value, filter)
        if callable(filter_result):  # Composite filter
            self._field_filters.append(filter_result)
        else:
            self._add_field_filter(*filter_result)
        return self

    def order_by(self, field_path: str, direction: Optional[str] = 'ASCENDING') -> 'Query':
        self.orders.append((field_path, direction))
        return self

    def limit(self, limit_amount: int) -> 'Query':
        self._limit = limit_amount
        return self

    def offset(self, offset_amount: int) -> 'Query':
        self._offset = offset_amount
        return self

    def start_at(self, document_fields_or_snapshot: Union[dict, DocumentSnapshot]) -> 'Query':
        self._start_at = (document_fields_or_snapshot, True)
        return self

    def start_after(self, document_fields_or_snapshot: Union[dict, DocumentSnapshot]) -> 'Query':
        self._start_at = (document_fields_or_snapshot, False)
        return self

    def end_at(self, document_fields_or_snapshot: Union[dict, DocumentSnapshot]) -> 'Query':
        self._end_at = (document_fields_or_snapshot, True)
        return self

    def end_before(self, document_fields_or_snapshot: Union[dict, DocumentSnapshot]) -> 'Query':
        self._end_at = (document_fields_or_snapshot, False)
        return self

    def _apply_cursor(
        self,
        document_fields_or_snapshot: Union[dict, DocumentSnapshot],
        doc_snapshot: Iterator[DocumentSnapshot],
        before: bool,
        start: bool,
    ) -> Iterator[DocumentSnapshot]:
        docs, doc_snapshot = tee(doc_snapshot)
        for idx, doc in enumerate(doc_snapshot):
            index = None
            if isinstance(document_fields_or_snapshot, dict):
                for k, v in document_fields_or_snapshot.items():
                    if doc.to_dict().get(k, None) == v:
                        index = idx
                    else:
                        index = None
                        break
            elif isinstance(document_fields_or_snapshot, DocumentSnapshot):
                if doc.id == document_fields_or_snapshot.id:
                    index = idx
            if index is not None:
                if before and start:
                    return islice(docs, index, None, None)
                elif not before and start:
                    return islice(docs, index + 1, None, None)
                elif before and not start:
                    return islice(docs, 0, index + 1, None)
                elif not before and not start:
                    return islice(docs, 0, index, None)

    @staticmethod
    def _compare_func(op: str) -> Callable[[T, T], bool]:
        if op == '==':
            return lambda x, y: x == y
        elif op == '!=':
            return lambda x, y: x != y
        elif op == '<':
            return lambda x, y: x < y
        elif op == '<=':
            return lambda x, y: x <= y
        elif op == '>':
            return lambda x, y: x > y
        elif op == '>=':
            return lambda x, y: x >= y
        elif op == 'in':
            return lambda x, y: x in y if y is not None else False
        elif op == 'array_contains':
            return lambda x, y: y in x if isinstance(x, (list, tuple)) else False
        elif op == 'array_contains_any':
            return (
                lambda x, y: any(val in y for val in x)
                if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple))
                else False
            )
        else:
            raise ValueError(f"Unsupported operator: {op}")


T = TypeVar("T")


class AsyncIterator(Generic[T]):
    def __init__(self, iterable: Iterator[T]):
        self._iter = iter(iterable)

    def __aiter__(self) -> "AsyncIterator[T]":
        return self

    async def __anext__(self) -> T:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class AsyncQuery(Query):
    def __init__(
        self,
        parent: 'AsyncCollectionReference',
        projection=None,
        field_filters=(),
        orders=(),
        limit=None,
        offset=None,
        start_at=None,
        end_at=None,
        all_descendants=False,
    ) -> None:
        self.parent = parent
        self.projection = projection
        self._field_filters = []
        self.orders = list(orders)
        self._limit = limit
        self._offset = offset
        self._start_at = start_at
        self._end_at = end_at
        self.all_descendants = all_descendants

        if field_filters:
            for field_filter in field_filters:
                # Support both tuple filters and composite callable filters (Or/And)
                if callable(field_filter):
                    self._field_filters.append(field_filter)
                else:
                    self._add_field_filter(*field_filter)

    async def _apply_cursor_async(
        self,
        document_fields_or_snapshot: Union[dict, AsyncDocumentSnapshot],
        doc_snapshots: List[AsyncDocumentSnapshot],
        before: bool,
        start: bool,
    ) -> List[AsyncDocumentSnapshot]:
        for idx, doc in enumerate(doc_snapshots):
            index = None
            if isinstance(document_fields_or_snapshot, dict):
                doc_dict = doc.to_dict()
                for k, v in document_fields_or_snapshot.items():
                    if doc_dict.get(k, None) == v:
                        index = idx
                    else:
                        index = None
                        break
            elif isinstance(document_fields_or_snapshot, AsyncDocumentSnapshot):
                if doc.id == document_fields_or_snapshot.id:
                    index = idx
            if index is not None:
                if before and start:
                    return doc_snapshots[index:]
                elif not before and start:
                    return doc_snapshots[index + 1 :]
                elif before and not start:
                    return doc_snapshots[: index + 1]
                elif not before and not start:
                    return doc_snapshots[:index]
        return doc_snapshots

    async def stream(self, transaction=None) -> AsyncGenerator[AsyncDocumentSnapshot, None]:
        doc_snapshots = []
        async for doc_snapshot in self.parent.stream():
            doc_snapshots.append(doc_snapshot)

        for filter in self._field_filters:
            if isinstance(filter, tuple):
                field, op, value = filter
                compare = self._compare_func(op)
                doc_snapshots = [
                    doc_snapshot
                    for doc_snapshot in doc_snapshots
                    if compare(doc_snapshot._get_by_field_path(field), value)
                ]
            else:  # Composite filter (And/Or)
                doc_snapshots = filter(doc_snapshots)

        if self.orders:
            for key, direction in self.orders:
                doc_snapshots_with_data = []
                for doc in doc_snapshots:
                    doc_dict = doc.to_dict()
                    doc_snapshots_with_data.append((doc, doc_dict.get(key, None)))

                doc_snapshots_with_data.sort(key=lambda x: x[1], reverse=direction == 'DESCENDING')
                doc_snapshots = [doc for doc, _ in doc_snapshots_with_data]

        if self._start_at:
            document_fields_or_snapshot, before = self._start_at
            doc_snapshots = await self._apply_cursor_async(document_fields_or_snapshot, doc_snapshots, before, True)

        if self._end_at:
            document_fields_or_snapshot, before = self._end_at
            doc_snapshots = await self._apply_cursor_async(document_fields_or_snapshot, doc_snapshots, before, False)

        if self._offset:
            doc_snapshots = islice(doc_snapshots, self._offset, None)

        if self._limit:
            doc_snapshots = islice(doc_snapshots, self._limit)

        # Yield the documents to make this an async generator
        for doc_snapshot in doc_snapshots:
            yield doc_snapshot
