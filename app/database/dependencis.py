from app.database.database import db
from sqlalchemy.orm import Session

def get_db():
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()
