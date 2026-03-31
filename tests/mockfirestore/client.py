from typing import AsyncGenerator, Iterable, Sequence

from mockfirestore.collection import AsyncCollectionReference, CollectionReference
from mockfirestore.document import AsyncDocumentReference, AsyncDocumentSnapshot, DocumentReference, DocumentSnapshot
from mockfirestore.transaction import AsyncTransaction, Transaction

# Shared data store for both sync and async MockFirestore instances
_shared_data = {}


class MockFirestore:
    def __init__(self) -> None:
        self._data = _shared_data

    def _ensure_path(self, path):
        current_position = self

        for el in path[:-1]:
            if type(current_position) in (MockFirestore, DocumentReference):
                current_position = current_position.collection(el)
            else:
                current_position = current_position.document(el)

        return current_position

    def document(self, path: str) -> DocumentReference:
        path = path.split("/")

        if len(path) % 2 != 0:
            raise Exception("Cannot create document at path {}".format(path))
        current_position = self._ensure_path(path)

        return current_position.document(path[-1])

    def collection(self, path: str) -> CollectionReference:
        path = path.split("/")

        if len(path) % 2 != 1:
            raise Exception("Cannot create collection at path {}".format(path))

        name = path[-1]
        if len(path) > 1:
            current_position = self._ensure_path(path)
            return current_position.collection(name)
        else:
            if name not in self._data:
                self._data[name] = {}
            return CollectionReference(self._data, [name])

    def collections(self) -> Sequence[CollectionReference]:
        return [CollectionReference(self._data, [collection_name]) for collection_name in self._data]

    def reset(self):
        global _shared_data
        _shared_data.clear()
        self._data = _shared_data

    def get_all(
        self,
        references: Iterable[DocumentReference],
        field_paths=None,
        transaction=None,
    ) -> Iterable[DocumentSnapshot]:
        for doc_ref in set(references):
            yield doc_ref.get()

    def transaction(self, **kwargs) -> Transaction:
        return Transaction(self, **kwargs)


class AsyncMockFirestore:
    def __init__(self) -> None:
        self._data = _shared_data

    def _ensure_path(self, path):
        current_position = self

        for el in path[:-1]:
            if type(current_position) in (MockFirestore, AsyncDocumentReference):
                current_position = current_position.collection(el)
            else:
                current_position = current_position.document(el)

        return current_position

    def document(self, path: str) -> AsyncDocumentReference:
        path = path.split("/")

        if len(path) % 2 != 0:
            raise Exception("Cannot create document at path {}".format(path))
        current_position = self._ensure_path(path)

        return current_position.document(path[-1])

    def collection(self, path: str) -> AsyncCollectionReference:
        path = path.split("/")

        if len(path) % 2 != 1:
            raise Exception("Cannot create collection at path {}".format(path))

        name = path[-1]
        if len(path) > 1:
            current_position = self._ensure_path(path)
            return current_position.collection(name)
        else:
            if name not in self._data:
                self._data[name] = {}
            return AsyncCollectionReference(self._data, [name])

    def collections(self) -> Sequence[AsyncCollectionReference]:
        return [AsyncCollectionReference(self._data, [collection_name]) for collection_name in self._data]

    async def reset(self):
        global _shared_data
        _shared_data.clear()
        self._data = _shared_data

    async def get_all(
        self,
        references: Iterable[AsyncDocumentReference],
        field_paths=None,
        transaction=None,
    ) -> AsyncGenerator[AsyncDocumentSnapshot, any]:
        for doc_ref in set(references):
            doc = await doc_ref.get()
            yield doc

    async def transaction(self, **kwargs) -> AsyncTransaction:
        return AsyncTransaction(self, **kwargs)
