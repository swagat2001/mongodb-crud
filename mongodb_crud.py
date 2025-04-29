"""
Production-grade MongoDB CRUD class with connection pooling, error handling, retry mechanisms,
logging, context manager support, type hints, and performance optimizations.

Author: swagat2001
"""

import logging
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.client_session import ClientSession
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult
from functools import wraps

T = TypeVar('T', bound=Dict[str, Any])

# Configure logging
logger = logging.getLogger("MongoDBCRUD")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def retry(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple = (errors.AutoReconnect, errors.NetworkTimeout, errors.ConnectionFailure)
):
    """
    Decorator for retrying a function with exponential backoff on specified exceptions.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
            logger.error(f"Operation failed after {max_retries} attempts.")
            raise
        return wrapper
    return decorator

class MongoDBCRUD(Generic[T]):
    """
    A production-grade MongoDB CRUD class with connection pooling, error handling,
    retry mechanisms, logging, context manager support, and performance optimizations.
    """

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str,
        max_pool_size: int = 100,
        min_pool_size: int = 0,
        server_selection_timeout_ms: int = 5000,
        connect_timeout_ms: int = 5000,
        socket_timeout_ms: int = 5000,
        **kwargs
    ):
        """
        Initialize the MongoDBCRUD instance.

        Args:
            uri (str): MongoDB connection URI.
            db_name (str): Database name.
            collection_name (str): Collection name.
            max_pool_size (int): Maximum number of connections in the pool.
            min_pool_size (int): Minimum number of connections in the pool.
            server_selection_timeout_ms (int): Server selection timeout in ms.
            connect_timeout_ms (int): Connection timeout in ms.
            socket_timeout_ms (int): Socket timeout in ms.
            **kwargs: Additional keyword arguments for MongoClient.
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self.session: Optional[ClientSession] = None
        self._client_kwargs = {
            "maxPoolSize": max_pool_size,
            "minPoolSize": min_pool_size,
            "serverSelectionTimeoutMS": server_selection_timeout_ms,
            "connectTimeoutMS": connect_timeout_ms,
            "socketTimeoutMS": socket_timeout_ms,
            **kwargs
        }
        self._connect()

    def _connect(self):
        """
        Establish a connection to MongoDB with connection pooling.
        """
        try:
            self.client = MongoClient(self.uri, **self._client_kwargs)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("MongoDB connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        Starts a session for transaction support.
        """
        if self.client is None:
            self._connect()
        self.session = self.client.start_session()
        self.session.start_transaction()
        logger.info("MongoDB session started with transaction.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and handle transaction commit/abort.
        """
        if self.session:
            try:
                if exc_type is None:
                    self.session.commit_transaction()
                    logger.info("Transaction committed.")
                else:
                    self.session.abort_transaction()
                    logger.warning("Transaction aborted due to exception.")
            except Exception as e:
                logger.error(f"Error during transaction handling: {e}")
            finally:
                self.session.end_session()
                logger.info("MongoDB session ended.")
                self.session = None

    @retry()
    def create_one(self, document: T) -> InsertOneResult:
        """
        Insert a single document into the collection.

        Args:
            document (T): The document to insert.

        Returns:
            InsertOneResult: The result of the insert operation.
        """
        try:
            result = self.collection.insert_one(document, session=self.session)
            logger.info(f"Inserted document with id: {result.inserted_id}")
            return result
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            raise

    @retry()
    def create_many(self, documents: List[T]) -> InsertManyResult:
        """
        Insert multiple documents into the collection.

        Args:
            documents (List[T]): The documents to insert.

        Returns:
            InsertManyResult: The result of the insert operation.
        """
        try:
            result = self.collection.insert_many(documents, session=self.session)
            logger.info(f"Inserted {len(result.inserted_ids)} documents.")
            return result
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise

    @retry()
    def read_one(self, filter: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> Optional[T]:
        """
        Find a single document matching the filter.

        Args:
            filter (Dict[str, Any]): The filter criteria.
            projection (Optional[Dict[str, Any]]): Fields to include or exclude.

        Returns:
            Optional[T]: The found document or None.
        """
        try:
            doc = self.collection.find_one(filter, projection, session=self.session)
            logger.info(f"Read one document: {doc}")
            return doc
        except Exception as e:
            logger.error(f"Error reading document: {e}")
            raise

    @retry()
    def read_many(self, filter: Dict[str, Any], projection: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[T]:
        """
        Find multiple documents matching the filter.

        Args:
            filter (Dict[str, Any]): The filter criteria.
            projection (Optional[Dict[str, Any]]): Fields to include or exclude.
            limit (int): Maximum number of documents to return.

        Returns:
            List[T]: List of found documents.
        """
        try:
            cursor = self.collection.find(filter, projection, session=self.session)
            if limit > 0:
                cursor = cursor.limit(limit)
            docs = list(cursor)
            logger.info(f"Read {len(docs)} documents.")
            return docs
        except Exception as e:
            logger.error(f"Error reading documents: {e}")
            raise

    @retry()
    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> UpdateResult:
        """
        Update a single document matching the filter.

        Args:
            filter (Dict[str, Any]): The filter criteria.
            update (Dict[str, Any]): The update operations.
            upsert (bool): Whether to insert if not found.

        Returns:
            UpdateResult: The result of the update operation.
        """
        try:
            result = self.collection.update_one(filter, update, upsert=upsert, session=self.session)
            logger.info(f"Updated {result.modified_count} document(s).")
            return result
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise

    @retry()
    def update_many(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> UpdateResult:
        """
        Update multiple documents matching the filter.

        Args:
            filter (Dict[str, Any]): The filter criteria.
            update (Dict[str, Any]): The update operations.
            upsert (bool): Whether to insert if not found.

        Returns:
            UpdateResult: The result of the update operation.
        """
        try:
            result = self.collection.update_many(filter, update, upsert=upsert, session=self.session)
            logger.info(f"Updated {result.modified_count} document(s).")
            return result
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise

    @retry()
    def delete_one(self, filter: Dict[str, Any]) -> DeleteResult:
        """
        Delete a single document matching the filter.

        Args:
            filter (Dict[str, Any]): The filter criteria.

        Returns:
            DeleteResult: The result of the delete operation.
        """
        try:
            result = self.collection.delete_one(filter, session=self.session)
            logger.info(f"Deleted {result.deleted_count} document(s).")
            return result
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    @retry()
    def delete_many(self, filter: Dict[str, Any]) -> DeleteResult:
        """
        Delete multiple documents matching the filter.

        Args:
            filter (Dict[str, Any]): The filter criteria.

        Returns:
            DeleteResult: The result of the delete operation.
        """
        try:
            result = self.collection.delete_many(filter, session=self.session)
            logger.info(f"Deleted {result.deleted_count} document(s).")
            return result
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def close(self):
        """
        Close the MongoDB client connection.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def __del__(self):
        self.close()
