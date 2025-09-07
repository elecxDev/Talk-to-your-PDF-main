from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def create_table():
    load_dotenv()
    db_url = os.getenv("SUPABASE_POSTGRES_URL")
    
    # Create engine
    engine = create_engine(db_url)
    
    # SQL commands
    commands = [
        "DROP TABLE IF EXISTS pdf_holder;",
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE TABLE pdf_holder (id SERIAL PRIMARY KEY, text TEXT, embedding VECTOR(1024));"
    ]
    
    # Execute commands
    with engine.connect() as conn:
        for command in commands:
            try:
                conn.execute(text(command))
                conn.commit()
                print(f"Successfully executed: {command}")
            except Exception as e:
                print(f"Error executing {command}: {str(e)}")

if __name__ == "__main__":
    create_table()
