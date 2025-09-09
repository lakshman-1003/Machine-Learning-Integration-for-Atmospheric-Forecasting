from src.models.db import Base, engine, SessionLocal
from src.models.user import User
from src.utils.auth import hash_password
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
        if not db.query(User).filter_by(email=email).first():
            u = User(email=email, name='Administrator', password_hash=hash_password(os.getenv('ADMIN_PASSWORD','admin123')), is_admin=True)
            db.add(u); db.commit(); print('Seeded admin user.')
        else:
            print('Admin already present.')

if __name__ == '__main__':
    main()
