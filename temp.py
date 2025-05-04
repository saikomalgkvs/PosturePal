import sqlite3

# Connect to the SQLite database
connection = sqlite3.connect('posture_data.db')

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Query to fetch all data from the database
cursor.execute("SELECT * FROM posture_data")  # Replace 'posture_data' with your table name

# Fetch and print all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
connection.close()

# # rewrite the values in the database for today's date to total time = 23, good time = 16, bad time = 7
# # Connect to the SQLite database
# connection = sqlite3.connect('posture_data.db')

# # Create a cursor object to interact with the database
# cursor = connection.cursor()

# # Today's date
# today = '2025-05-03'

# # Update query to rewrite the values for today's date
# cursor.execute("""
#     UPDATE posture_data
#     SET total_monitoring_time = ?, good_posture_duration = ?, bad_posture_duration = ?
#     WHERE date = ?
# """, (63, 47, 14, today))

# # Commit the changes and close the connection
# connection.commit()
# connection.close()
