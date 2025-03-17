import sqlite3

'''materials_data = {
    "FR-4": [0.2, 0.5, 0.7, 1.0, 1.1, 1.8, 1.9],
    "Rogers RO4003C": [0.1, 0.3, 0.4, 0.6, 0.65, 1.1, 1.2],
    "Rogers RO4350B": [0.12, 0.35, 0.5, 0.75, 0.8, 1.3, 1.4],
    "Isola I-Tera MT40": [0.11, 0.33, 0.45, 0.7, 0.75, 1.2, 1.3],
    "Nelco N4000-13": [0.15, 0.45, 0.6, 0.9, 1.0, 1.6, 1.7],
    "Teflon (PTFE)": [0.05, 0.15, 0.2, 0.3, 0.35, 0.6, 0.65],
    "Megtron 6": [0.08, 0.25, 0.35, 0.5, 0.55, 0.9, 1.0]
}'''
# The below material was made using the script stripline_calc_loss_db.py
# W = 6mil
materials_data = {
    "FR-4": [0.322,1.488,2.445,3.364,4.009,5.871],
    "Rogers RO4003C": [0.2477,0.8703,1.435,1.99,2.3753,3.3626],
    "Rogers RO4350B": [0.2486,0.8786,1.4546,2.0165,2.4126,3.4091],
    "Isola I-Tera MT40": [0.2485,0.8768,1.4506,2.0102,2.4051,3.4037],
    "Nelco N4000-13": [0.253,0.956,1.5866,2.1632,2.5566,3.5271],
    "Teflon (PTFE)": [0.2379,0.714,1.0757,1.4301,1.6934,2.4233],
    "Megtron 6": [0.2444,0.8004,1.2843,1.7793,2.1416,3.1058]
}

frequencies = [1, 10, 16, 25, 28, 53, 56]

conn = sqlite3.connect("materials.db")
cursor = conn.cursor()

# Create materials table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS materials (
        material_id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT UNIQUE
    )
''')

# Create losses table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS losses (
        material_id INTEGER,
        frequency_GHz REAL,
        loss_dB_inch REAL,
        FOREIGN KEY (material_id) REFERENCES materials(material_id),
        PRIMARY KEY (material_id, frequency_GHz)
    )
''')

# Insert materials
for material in materials_data.keys():
    cursor.execute("INSERT OR IGNORE INTO materials (material_name) VALUES (?)", (material,))

# Insert loss data
for material, losses in materials_data.items():
    cursor.execute("SELECT material_id FROM materials WHERE material_name = ?", (material,))
    material_id = cursor.fetchone()[0]
    for freq, loss in zip(frequencies, losses):
        cursor.execute("INSERT OR REPLACE INTO losses (material_id, frequency_GHz, loss_dB_inch) VALUES (?, ?, ?)",
                       (material_id, freq, loss))

conn.commit()
conn.close()

print("Database 'materials.db' created successfully with alternative structure!")