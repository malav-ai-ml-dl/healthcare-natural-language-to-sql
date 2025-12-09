import sqlite3

# Create new healthcare database
conn = sqlite3.connect("healthcare.db")
cursor = conn.cursor()

# Drop tables if they already exist
cursor.execute("DROP TABLE IF EXISTS patients")
cursor.execute("DROP TABLE IF EXISTS visits")
cursor.execute("DROP TABLE IF EXISTS medications")

# Patients table
cursor.execute("""
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    condition TEXT
)
""")

# Visits table
cursor.execute("""
CREATE TABLE visits (
    visit_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    visit_date TEXT,
    doctor TEXT,
    diagnosis TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
)
""")

# Medications table
cursor.execute("""
CREATE TABLE medications (
    med_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    medication TEXT,
    dosage TEXT,
    start_date TEXT,
    end_date TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
)
""")

# Insert sample patients
cursor.executemany("INSERT INTO patients (name, age, gender, condition) VALUES (?, ?, ?, ?)", [
    ("Alice Smith", 34, "F", "Diabetes"),
    ("Bob Johnson", 45, "M", "Hypertension"),
    ("Charlie Lee", 29, "M", "Asthma")
])

# Insert sample visits
cursor.executemany("INSERT INTO visits (patient_id, visit_date, doctor, diagnosis) VALUES (?, ?, ?, ?)", [
    (1, "2025-01-15", "Dr. Adams", "Routine Checkup"),
    (2, "2025-02-20", "Dr. Brown", "Blood Pressure Monitoring"),
    (3, "2025-03-05", "Dr. Clark", "Asthma Attack")
])

# Insert sample medications
cursor.executemany("INSERT INTO medications (patient_id, medication, dosage, start_date, end_date) VALUES (?, ?, ?, ?, ?)", [
    (1, "Metformin", "500mg", "2025-01-15", "2025-12-31"),
    (2, "Lisinopril", "10mg", "2025-02-20", "2025-12-31"),
    (3, "Albuterol", "2 puffs", "2025-03-05", "2025-12-31")
])

conn.commit()
conn.close()
print("✅ healthcare.db created with sample data")
