# MongoDB CRUD Utility

A production-grade, reusable Python class for MongoDB CRUD operations with robust error handling, connection pooling, retry mechanisms, logging, context manager support, type hints, and performance optimizations.

## Features

- **Connection Pooling** for high performance and scalability
- **Comprehensive Error Handling** for reliability
- **Retry Mechanism** with exponential backoff for transient failures
- **Transaction Support** via context manager
- **Proper Logging** for observability
- **Type Hints** and full documentation
- **Optimized Connection Settings** for production workloads

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/swagat2001/mongodb-crud.git
   cd mongodb-crud
   ```

2. **Install dependencies:**
   ```sh
   pip install pymongo
   ```

## Usage

### Import the Class

```python
from mongodb_crud import MongoDBCRUD
```

### Initialize

```python
crud = MongoDBCRUD(
    uri="mongodb://localhost:27017",
    db_name="your_db",
    collection_name="your_collection"
)
```

### Basic CRUD Operations

```python
# Create
crud.create_one({"name": "Alice", "age": 30})
crud.create_many([{"name": "Bob"}, {"name": "Charlie"}])

# Read
doc = crud.read_one({"name": "Alice"})
docs = crud.read_many({"age": {"$gte": 18}}, limit=10)

# Update
crud.update_one({"name": "Alice"}, {"$set": {"age": 31}})
crud.update_many({"age": {"$lt": 18}}, {"$set": {"minor": True}})

# Delete
crud.delete_one({"name": "Bob"})
crud.delete_many({"minor": True})
```

### Transaction Support (Context Manager)

```python
with MongoDBCRUD(
    uri="mongodb://localhost:27017",
    db_name="your_db",
    collection_name="your_collection"
) as crud:
    crud.create_one({"name": "Transactional User"})
    crud.update_one({"name": "Transactional User"}, {"$set": {"age": 25}})
    # All operations in this block are part of a transaction
```

### Closing the Connection

```python
crud.close()
```

## Advanced Options

You can customize connection pooling and timeouts:

```python
crud = MongoDBCRUD(
    uri="mongodb://localhost:27017",
    db_name="your_db",
    collection_name="your_collection",
    max_pool_size=200,
    min_pool_size=10,
    server_selection_timeout_ms=10000,
    connect_timeout_ms=10000,
    socket_timeout_ms=10000
)
```

## Logging

Logs are output to the console by default. You can further configure logging as needed.

## License

MIT License

---

**Author:** [swagat2001](https://github.com/swagat2001)
