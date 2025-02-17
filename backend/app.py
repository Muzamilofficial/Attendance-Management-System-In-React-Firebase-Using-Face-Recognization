from flask import Flask, jsonify, send_file, request
import firebase_admin
from firebase_admin import credentials, db, storage
import pandas as pd
import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import pickle
from flask_cors import CORS
import base64
from threading import Lock
from collections import defaultdict

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Firebase
cred = credentials.Certificate("backend/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-system-realtime-default-rtdb.firebaseio.com/",
    'storageBucket': "attendance-system-realtime.appspot.com"
})

# Load face encodings
with open("backend/Encode.p", "rb") as file:
    encodeListKnownWithIDs = pickle.load(file)
encodeListKnown, studentIDs = encodeListKnownWithIDs

class AttendanceSession:
    def __init__(self):
        self.active = True
        self.start_time = datetime.now()
        self.detected_students = set()  # Store student IDs that have been detected
        self.processed_faces = {}  # Dictionary to store face encodings and their last detection time
        self.detection_cooldown = 300  # 5 minutes cooldown before processing the same face again

processing_sessions = {}
attendance_lock = Lock()

# Routes
@app.route('/')
def index():
    return "Attendance System Backend"

@app.route('/get_courses/<major>/<section>')
def get_courses(major, section):
    try:
        ref = db.reference(f"Majors/{major}/Sections/{section}/Students")
        students = ref.get()
        
        if not students:
            return jsonify([])
        
        courses = set()
        for student_id, student_data in students.items():
            if isinstance(student_data, dict) and 'Courses' in student_data:
                courses.update(student_data['Courses'].keys())
        
        print(f"Fetched courses for {major}/{section}: {list(courses)}")  # Debugging
        return jsonify(list(courses))
    except Exception as e:
        print(f"Error in /get_courses: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_attendance/<major>/<section>/<course>')
def start_attendance(major, section, course):
    try:
        session_id = f"{major}_{section}_{course}"
        processing_sessions[session_id] = AttendanceSession()
        return jsonify({
            "status": "success",
            "message": "Attendance session started",
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error starting attendance: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to start attendance: {str(e)}"
        }), 500

@app.route('/stop_attendance/<major>/<section>/<course>')
def stop_attendance(major, section, course):
    try:
        session_id = f"{major}_{section}_{course}"
        if session_id not in processing_sessions:
            return jsonify({"error": "No active session found"}), 404

        session = processing_sessions[session_id]
        session.active = False

        # Mark attendance in Firebase
        with attendance_lock:
            current_time = datetime.now()
            for student_id in session.detected_students:
                try:
                    # Update attendance in Firebase
                    student_ref = db.reference(f"Majors/{major}/Sections/{section}/Students/{student_id}/Courses/{course}")
                    student_data = student_ref.get() or {}
                    
                    # Increment attendance count
                    attendance_count = student_data.get('count', 0) + 1
                    
                    # Update Firebase
                    student_ref.update({
                        "count": attendance_count,
                        "last_marked": current_time.isoformat()
                    })
                except Exception as e:
                    print(f"Error marking attendance for student {student_id}: {e}")
                    continue

        # Clean up session
        session_data = {
            "total_students_marked": len(session.detected_students),
            "session_duration": (current_time - session.start_time).total_seconds() / 60  # in minutes
        }
        del processing_sessions[session_id]

        return jsonify({
            "status": "success",
            "message": "Attendance stopped successfully",
            "data": session_data
        })

    except Exception as e:
        print(f"Error in stop_attendance: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to stop attendance: {str(e)}"
        }), 500

@app.route('/check_attendance_status/<major>/<section>/<course>')
def check_attendance_status(major, section, course):
    try:
        current_time = datetime.now()
        ref = db.reference(f"Majors/{major}/Sections/{section}/Students")
        students = ref.get()
        
        if not students:
            return jsonify({"status": "error", "message": "No students found"})
            
        recently_marked = []
        for student_id, student_data in students.items():
            if 'Courses' in student_data and course in student_data['Courses']:
                last_marked_str = student_data['Courses'][course].get('last_marked')
                if last_marked_str:
                    last_marked = datetime.fromisoformat(last_marked_str)
                    time_diff = current_time - last_marked
                    if time_diff.total_seconds() < 8 * 3600:  # 8 hours in seconds
                        recently_marked.append({
                            'student_id': student_id,
                            'name': student_data.get('Name', 'Unknown'),
                            'last_marked': last_marked_str
                        })
        
        return jsonify({
            "status": "success",
            "recently_marked": recently_marked
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Extract data from request
        data = request.json
        frame_data = base64.b64decode(data['frame'])
        major = data['major']
        section = data['section']
        course = data['course']
        
        # Create session ID and validate session
        session_id = f"{major}_{section}_{course}"
        session = processing_sessions.get(session_id)
        
        if not session or not session.active:
            return jsonify({"error": "No active session"}), 400

        # Convert frame data to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get enrolled students
        enrolled_students = get_enrolled_students_data(major, section, course)
        
        # Process frame with face recognition
        processed_frame, newly_detected = process_frame_with_recognition(
            frame, 
            enrolled_students, 
            encodeListKnown, 
            studentIDs, 
            course,
            session
        )
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "processedFrame": processed_frame_b64,
            "detectedStudents": list(newly_detected),
            "totalDetected": len(session.detected_students)
        })

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_excel/<major>/<section>/<course>')
def download_excel(major, section, course):
    try:
        # Get attendance data
        ref = db.reference(f"Majors/{major}/Sections/{section}/Students")
        students_data = ref.get()
        
        if not students_data:
            return jsonify({"error": "No student data found"}), 404
        
        excel_data = []
        for student_id, student_data in students_data.items():
            if isinstance(student_data, dict) and 'Courses' in student_data:
                course_data = student_data['Courses'].get(course, {})
                if course_data:  # Only include students enrolled in the course
                    excel_data.append({
                        'Student ID': student_id,
                        'Name': student_data.get('Name', 'N/A'),
                        'Attendance Count': course_data.get('count', 0),
                        'Last Marked': course_data.get('last_marked', 'Never')
                    })
        
        if not excel_data:
            return jsonify({"error": "No attendance data found for this course"}), 404
            
        # Create DataFrame and sort by Student ID
        df = pd.DataFrame(excel_data)
        df = df.sort_values('Student ID')
        
        # Ensure downloads directory exists
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Generate Excel file with timestamp
        filename = f'attendance_{major}_{section}_{course}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        excel_path = os.path.join(downloads_dir, filename)
        
        # Create Excel with formatting
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Add header format
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            # Format columns
            for col_num, value in enumerate(df.columns.values):
                max_length = max(
                    df[value].astype(str).apply(len).max(),
                    len(str(value))
                ) + 2
                worksheet.set_column(col_num, col_num, max_length)
                worksheet.write(0, col_num, value, header_format)
        
        # Send file and then remove it
        response = send_file(
            excel_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        # Delete file after sending
        @response.call_on_close
        def cleanup():
            try:
                os.remove(excel_path)
            except Exception as e:
                print(f"Error cleaning up file {excel_path}: {e}")
                
        return response
        
    except Exception as e:
        print(f"Error in download_excel: {e}")  # Log the actual error
        return jsonify({"error": "Failed to generate Excel file", "details": str(e)}), 500

# Helper functions remain the same...
def process_frame_with_recognition(frame, enrolled_students, encode_list_known, student_ids, current_course, session):
    """
    Process frame for face recognition with tracking and cooldown
    """
    newly_detected = set()
    current_time = datetime.now()
    
    # Resize frame for faster processing
    imgS = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(imgS)
    
    if face_locations:
        face_encodings = face_recognition.face_encodings(imgS, face_locations)
        
        for encoding, location in zip(face_encodings, face_locations):
            # Convert encoding to bytes for hashing
            encoding_bytes = encoding.tobytes()
            
            # Check if we've recently processed this face
            if encoding_bytes in session.processed_faces:
                last_detection_time = session.processed_faces[encoding_bytes]
                if (current_time - last_detection_time).total_seconds() < session.detection_cooldown:
                    # Skip processing this face if within cooldown period
                    continue
            
            matches = face_recognition.compare_faces(encode_list_known, encoding, tolerance=0.6)
            
            top, right, bottom, left = [coord * 4 for coord in location]
            
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(encode_list_known, encoding))
                student_id = student_ids[best_match_index]
                
                if student_id in enrolled_students:
                    student_data = enrolled_students[student_id]
                    last_marked_str = student_data.get('Courses', {}).get(current_course, {}).get('last_marked')
                    
                    # Check if student was marked in last 8 hours
                    is_recent = False
                    if last_marked_str:
                        last_marked = datetime.fromisoformat(last_marked_str)
                        time_diff = current_time - last_marked
                        is_recent = time_diff.total_seconds() < 8 * 3600  # 8 hours
                    
                    # Update tracking
                    session.processed_faces[encoding_bytes] = current_time
                    if not is_recent and student_id not in session.detected_students:
                        session.detected_students.add(student_id)
                        newly_detected.add(student_id)
                    
                    # Set color based on whether attendance can be marked
                    box_color = (0, 128, 255) if is_recent else (0, 255, 0)
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                    
                    name_text = student_data.get('Name', student_id)
                    status_text = "Already Marked" if is_recent else "Attendance Marked"
                    display_text = f"{name_text} - {status_text}"
                    
                    cv2.putText(frame, display_text, 
                              (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 
                              0.6, (255, 255, 255), 2)
                else:
                    # Draw red box for non-enrolled students
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Not Enrolled", (left + 6, bottom - 6), 
                              cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Draw red box for unknown faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left + 6, bottom - 6), 
                          cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, newly_detected

def get_enrolled_students_data(major, section, course):
    """Helper function to get enrolled students data from Firebase"""
    try:
        ref = db.reference(f"Majors/{major}/Sections/{section}/Students")
        all_students = ref.get()
        
        if not all_students:
            return {}
        
        return {
            student_id: student_data
            for student_id, student_data in all_students.items()
            if isinstance(student_data, dict) and 
            'Courses' in student_data and 
            course in student_data['Courses']
        }
    except Exception as e:
        print(f"Error getting enrolled students data: {e}")
        return {}

if __name__ == '__main__':
    # Run the app on all available IPs (0.0.0.0) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)