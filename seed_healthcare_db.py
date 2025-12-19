import sqlite3
import random
from datetime import datetime, timedelta

# Connect to DB
conn = sqlite3.connect("healthcare.db")
cursor = conn.cursor()

# Drop old tables
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

# ------------------ Data Pools ------------------
first_names = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia"]
last_names = ["Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas", "Moore", "Martin"]

visit_reasons = [
    "Routine checkup", "Diabetes follow-up", "Blood pressure monitoring",
    "Flu symptoms", "Chest pain", "Annual physical",
    "Asthma review", "Medication refill", "Post-surgery review"
]

medication_list = [
    "Metformin", "Insulin", "Aspirin", "Atorvastatin",
    "Lisinopril", "Amoxicillin", "Ibuprofen",
    "Albuterol", "Omeprazole"
]

# ------------------ Insert Patients ------------------
num_patients = 200  # 🔹 increase this to 500 / 1000 if needed
patients = []

for _ in range(num_patients):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(18, 85)
    gender = random.choice(["M", "F"])
    patients.append((name, age, gender))

cursor.executemany(
    "INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)",
    patients
)

# ------------------ Insert Visits ------------------
visits = []
for patient_id in range(1, num_patients + 1):
    for _ in range(random.randint(1, 5)):
        visit_date = datetime.now() - timedelta(days=random.randint(1, 1000))
        reason = random.choice(visit_reasons)
        visits.append((
            patient_id,
            visit_date.strftime("%Y-%m-%d"),
            reason
        ))

cursor.executemany(
    "INSERT INTO visits (patient_id, visit_date, reason) VALUES (?, ?, ?)",
    visits
)

# ------------------ Insert Medications ------------------
medications = []
for patient_id in range(1, num_patients + 1):
    for med in random.sample(medication_list, random.randint(1, 3)):
        medications.append((patient_id, med))

cursor.executemany(
    "INSERT INTO medications (patient_id, medication) VALUES (?, ?)",
    medications
)

# Commit & close
conn.commit()
conn.close()

print("✅ healthcare.db created with realistic large healthcare dataset")
