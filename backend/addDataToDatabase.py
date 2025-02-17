import firebase_admin
from firebase_admin import credentials, db
import csv
import re

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-system-realtime-default-rtdb.firebaseio.com/"
})

# Get a reference to the "Majors" node in the database
ref = db.reference("Majors")

# Read the CSV file
csv_file_path = "app/Untitled form.csv"
data = {
    "CS": {
        "Sections": {
            "A": {"Students": {}},
            "B": {"Students": {}}
        }
    },
    "SE": {
        "Sections": {
            "A": {"Students": {}},
            "B": {"Students": {}}
        }
    }
}

# Function to extract only numbers from the seat number
def extract_numbers(seat_number):
    return re.sub(r'\D', '', seat_number)  # Remove all non-numeric characters

with open(csv_file_path, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        full_name = row["Full Name as mentioned in the University Documents"]
        seat_number = extract_numbers(row["Seat Number?"])  # Extract only numbers
        discipline = row["Discipline?"]
        section = row["Section?"]

        # Determine the major based on discipline
        major = "CS" if discipline == "Computer Science" else "SE"

        # Add student data to the appropriate section
        data[major]["Sections"][section]["Students"][seat_number] = {
            "Name": full_name,
            "Courses": {
                "BSCS 604": {
                    "count": 0,
                    "last_marked": "2024-12-03T08:30:00"
                },
                "BSCS 606": {
                    "count": 0,
                    "last_marked": "2024-12-02T14:00:00"
                }
            }
        }

# Setting data to Firebase
for major, major_data in data.items():
    ref.child(major).set(major_data)

print("Data has been successfully uploaded to Firebase.")