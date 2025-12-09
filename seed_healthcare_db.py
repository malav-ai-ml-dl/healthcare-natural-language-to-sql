import sqlite3

# Connect (will create file if it doesn’t exist)
conn = sqlite3.connect("healthcare.db")
cursor = conn.cursor()

# Drop old tables (safety for re-run)
cursor.execute("DROP TABLE IF EXISTS patients")
cursor.execute("DROP TABLE IF EXISTS visits")
cursor.execute("DROP TABLE IF EXISTS medications")

# Create tables
cursor.execute("""
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT
)
""")

cursor.execute("""
CREATE TABLE visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    visit_date TEXT,
    reason TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

cursor.execute("""
CREATE TABLE medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    medication TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

# Insert sample patients
patients = [
    ("Alice Smith", 72, "F"),
    ("Bob Johnson", 65, "M"),
    ("Charlie Brown", 45, "M"),
    ("Diana King", 55, "F"),
]
cursor.executemany("INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)", patients)

# Insert visits
visits = [
    (1, "2024-02-15", "Diabetes checkup"),
    (2, "2024-06-10", "Routine examination"),
    (3, "2025-01-20", "Flu symptoms"),
    (1, "2025-03-05", "Blood pressure follow-up"),
]
cursor.executemany("INSERT INTO visits (patient_id, visit_date, reason) VALUES (?, ?, ?)", visits)

# Insert medications
medications = [
    (1, "Metformin"),
    (1, "Insulin"),
    (2, "Aspirin"),
    (3, "Amoxicillin"),
    (4, "Atorvastatin"),
]
cursor.executemany("INSERT INTO medications (patient_id, medication) VALUES (?, ?)", medications)

# Save and close
conn.commit()
conn.close()

print("✅ healthcare.db created and populated with sample data!")
