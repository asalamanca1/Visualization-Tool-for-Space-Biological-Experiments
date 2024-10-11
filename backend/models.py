from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.types import JSON

db = SQLAlchemy()

class Experiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String, nullable=False)
    # csv_data_path = db.Column(db.String, nullable=False)
    text_content = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
