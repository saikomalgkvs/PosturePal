import sqlite3

# Connect to the SQLite database
connection = sqlite3.connect('posture_data.db')

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Query to fetch all data from the database
# cursor.execute("SELECT * FROM posture_data")  # Replace 'posture_data' with your table name

# Fetch and print all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
connection.close()