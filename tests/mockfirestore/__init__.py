# try to import gcloud exceptions
# and if gcloud is not installed, define our own
try:
    from google.api_core.exceptions import AlreadyExists, ClientError, Conflict, NotFound
except ImportError:
    from mockfirestore.exceptions import ClientError, Conflict, NotFound, AlreadyExists

from mockfirestore._helpers import Timestamp
from mockfirestore.client import AsyncMockFirestore, MockFirestore
from mockfirestore.collection import CollectionReference
from mockfirestore.document import DocumentReference, DocumentSnapshot
from mockfirestore.query import Query
from mockfirestore.transaction import Transaction
