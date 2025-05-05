import sqlite3

connection = sqlite3.connect('posture_data.db')
cursor = connection.cursor()
# cursor.execute("SELECT * FROM posture_data")  # Replace 'posture_data' with your table name
# delete todays row
# cursor.execute("DELETE FROM posture_data WHERE date = date('now')")  # Replace 'posture_data' with your table name
cursor.execute("SELECT * FROM posture_data")  # Replace 'posture_data' with your table name

rows = cursor.fetchall()
for row in rows:
    print(row)

# Commit the changes and close the connection
connection.commit()

connection.close()


