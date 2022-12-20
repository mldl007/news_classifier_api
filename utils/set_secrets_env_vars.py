from utils.json_parser import JSONParser
import os
from logger.logger import MongoLogger


def set_db_secrets_env():
    """
    Function to set Mongo DB secrets as environment variables.
    """
    logger = MongoLogger()
    try:
        if os.path.exists(os.path.join(".", "secrets", "secrets.json")):
            json_parser = JSONParser(os.path.join(".", "secrets", "secrets.json"))
            db_secrets_dict = json_parser.parse_json()
            db_url = db_secrets_dict['db_url']
            os.environ['DB_URL'] = db_url
    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected set_secrets_env error: {e}")
        raise
