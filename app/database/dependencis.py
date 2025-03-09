from app.database.database import db

def get_db():
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()
