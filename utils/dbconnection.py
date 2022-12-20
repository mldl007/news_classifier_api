import pymongo
from logger.logger import MongoLogger


class DBConnection:
    """
    returns connection object of the specified MongoDB database.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_connection = None

    def connect(self):  # method to return connection object
        logger = MongoLogger()
        try:
            self.db_connection = pymongo.MongoClient(self.db_url)
            _ = self.db_connection.list_database_names()
        except Exception as conn_exception:
            self.db_connection = None
            logger.log_to_db(level="CRITICAL",
                             message=f"DB connection error: {conn_exception}")
            raise
        else:
            logger.log_to_db(level="INFO",
                             message="DB SERVER CONNECTION SUCCESSFUL")
        return self.db_connection
