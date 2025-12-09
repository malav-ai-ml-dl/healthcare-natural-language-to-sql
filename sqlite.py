import sqlite3

## connect to sqllite
connection=sqlite3.connect("students.db")

##create a cursor object to insert record,create table
cursor=connection.cursor()

## create the table
table_info="""
create table STUDENTS(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

## Insert some more records
cursor.execute('''Insert Into STUDENTS values('Mukesh','Data Science','A',86)''')
cursor.execute('''Insert Into STUDENTS values('Jacob','DEVOPS','A',50)''')
cursor.execute('''Insert Into STUDENTS values('Krish','Data Science','A',90)''')
cursor.execute('''Insert Into STUDENTS values('John','Data Science','B',100)''')
cursor.execute('''Insert Into STUDENTS values('Dipesh','DEVOPS','A',35)''')

## Display all the records
print("The inserted records are")
data=cursor.execute('''Select * from STUDENTS''')
for row in data:
    print(row)

## Commit your changes in the database
connection.commit()
connection.close()
