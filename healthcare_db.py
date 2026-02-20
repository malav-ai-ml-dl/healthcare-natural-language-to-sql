import sqlite3
import random
from datetime import datetime, timedelta

# -----------------------------------------------------
# Create Database
# -----------------------------------------------------
conn = sqlite3.connect("demo_healthcare.db")
cursor = conn.cursor()

cursor.execute("PRAGMA foreign_keys = ON;")

# -----------------------------------------------------
# Create Tables
# -----------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    city TEXT,
    chronic_condition TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS visits(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    visit_date TEXT,
    department TEXT,
    reason TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS medications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    medication TEXT,
    dosage TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS lab_results(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    test_name TEXT,
    result_value REAL,
    unit TEXT,
    test_date TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

# -----------------------------------------------------
# Seed Data
# -----------------------------------------------------
first_names_male = ["Aarav", "Vihaan", "Arjun", "Rohan", "Aditya", "Kabir", "Ishaan", "Rahul", "Karan", "Manish"]
first_names_female = ["Aanya", "Diya", "Ira", "Ananya", "Priya", "Sneha", "Kavya", "Meera", "Riya", "Pooja"]
last_names = ["Sharma", "Patel", "Verma", "Reddy", "Nair", "Singh", "Gupta", "Joshi", "Mehta", "Iyer"]

cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Kolkata"]
chronic_conditions = ["Diabetes", "Hypertension", "Asthma", "None"]

departments = ["Cardiology", "General Medicine", "Endocrinology", "Pulmonology", "Orthopedics"]
reasons = ["Routine Checkup", "Follow-up", "Chest Pain", "High Sugar", "Breathing Issue", "Joint Pain"]

medication_list = {
    "Diabetes": ["Metformin", "Insulin"],
    "Hypertension": ["Amlodipine", "Losartan"],
    "Asthma": ["Salbutamol", "Budesonide"],
    "None": ["Vitamin D", "Paracetamol"]
}

lab_tests = {
    "HbA1c": ("%", lambda: round(random.uniform(4.5, 9.5), 1)),
    "Blood Glucose": ("mg/dL", lambda: round(random.uniform(70, 200), 1)),
    "Cholesterol": ("mg/dL", lambda: round(random.uniform(150, 280), 1)),
    "Blood Pressure": ("mmHg", lambda: round(random.uniform(80, 160), 1)),
    "Hemoglobin": ("g/dL", lambda: round(random.uniform(10, 18), 1))
}

num_patients = random.randint(50, 100)

for _ in range(num_patients):
    gender = random.choice(["M", "F"])
    first_name = random.choice(first_names_male if gender == "M" else first_names_female)
    last_name = random.choice(last_names)
    name = f"{first_name} {last_name}"
    age = random.randint(18, 85)
    city = random.choice(cities)
    chronic = random.choices(chronic_conditions, weights=[0.3, 0.3, 0.2, 0.2])[0]

    cursor.execute("""
    INSERT INTO patients (name, age, gender, city, chronic_condition)
    VALUES (?, ?, ?, ?, ?)
    """, (name, age, gender, city, chronic))

    patient_id = cursor.lastrowid

    # Visits
    for _ in range(random.randint(2, 5)):
        visit_date = datetime.now() - timedelta(days=random.randint(0, 365))
        department = random.choice(departments)
        reason = random.choice(reasons)
        cursor.execute("""
        INSERT INTO visits (patient_id, visit_date, department, reason)
        VALUES (?, ?, ?, ?)
        """, (patient_id, visit_date.strftime("%Y-%m-%d"), department, reason))

    # Medications
    meds = medication_list[chronic]
    for _ in range(random.randint(1, 4)):
        med = random.choice(meds)
        dosage = random.choice(["5mg", "10mg", "20mg", "1 tablet daily", "2 times daily"])
        cursor.execute("""
        INSERT INTO medications (patient_id, medication, dosage)
        VALUES (?, ?, ?)
        """, (patient_id, med, dosage))

    # Lab Results
    for _ in range(random.randint(2, 6)):
        test_name = random.choice(list(lab_tests.keys()))
        unit, value_func = lab_tests[test_name]
        result_value = value_func()
        test_date = datetime.now() - timedelta(days=random.randint(0, 365))
        cursor.execute("""
        INSERT INTO lab_results (patient_id, test_name, result_value, unit, test_date)
        VALUES (?, ?, ?, ?, ?)
        """, (patient_id, test_name, result_value, unit, test_date.strftime("%Y-%m-%d")))

# -----------------------------------------------------
# Commit & Close
# -----------------------------------------------------
conn.commit()
conn.close()

print("Database created successfully")
