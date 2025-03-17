import sqlite3

conn = sqlite3.connect("materials.db")
cursor = conn.cursor()

# Create cables table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS cables (
        cable_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cable_name TEXT UNIQUE
    )
""")

# Create cable_losses table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS cable_losses (
        cable_loss_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cable_id INTEGER,
        frequency_GHz REAL,
        loss_dB_meter REAL,
        FOREIGN KEY (cable_id) REFERENCES cables(cable_id)
    )
""")

# Insert cable types
cable_types = [
    "twinax_AWG26","twinax_AWG28","twinax_AWG30","twinax_AWG32"
]
for cable in cable_types:
    cursor.execute("INSERT OR IGNORE INTO cables (cable_name) VALUES (?)", (cable,))

# Insert loss data
loss_data = [
    (2.5, 1.25, 1.6, 1.9,2.6),
    (5, 1.8, 2.2, 2.5, 3.8),
    (10, 2.9, 3.25, 3.7, 5.4),
    (15, 3.65, 4.1, 4.7, 5.2),
    (18, 4.15, 5, 5.2, 7.2)
]

for freq, *losses in loss_data:
    for i, loss in enumerate(losses):
        cable_name = cable_types[i]
        cursor.execute("SELECT cable_id FROM cables WHERE cable_name = ?", (cable_name,))
        cable_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO cable_losses (cable_id, frequency_GHz, loss_dB_meter) VALUES (?, ?, ?)",
                       (cable_id, freq, loss))

conn.commit()
conn.close()
print("Cable database created and populated.")