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
    "AWG 24", "AWG 26", "AWG 28", "AWG 30",
    "DAC AWG 24", "DAC AWG 26", "DAC AWG 28", "DAC AWG 30"
]
for cable in cable_types:
    cursor.execute("INSERT OR IGNORE INTO cables (cable_name) VALUES (?)", (cable,))

# Insert loss data
loss_data = [
    (1, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6),
    (10, 1, 1.3, 1.6, 2, 1.2, 1.5, 1.8, 2.2),
    (16, 1.5, 1.9, 2.3, 2.8, 1.8, 2.2, 2.6, 3.1),
    (25, 2.2, 2.7, 3.3, 4, 2.5, 3, 3.6, 4.3),
    (28, 2.5, 3, 3.7, 4.5, 2.8, 3.3, 4, 4.8),
    (53, 4, 4.9, 6, 7.2, 4.5, 5.4, 6.5, 7.8),
    (56, 4.3, 5.2, 6.3, 7.6, 4.8, 5.7, 6.9, 8.2)
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