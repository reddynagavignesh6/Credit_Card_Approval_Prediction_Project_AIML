import sqlite3

connection = sqlite3.connect("credit_card.db")

cursor = connection.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS predictions(

id INTEGER PRIMARY KEY AUTOINCREMENT,

applicant_name TEXT,

prediction TEXT,

confidence REAL,

prediction_date TEXT,

prediction_time TEXT

)

""")

connection.commit()

connection.close()

print("Database Created Successfully")