import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";

// Set the base URL for Axios using environment variable
// axios.defaults.baseURL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:5000";
axios.defaults.baseURL = "http://192.168.100.19:5000";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/major" element={<MajorScreen />} />
        <Route path="/section" element={<SectionScreen />} />
        <Route path="/course" element={<CourseScreen />} />
        <Route path="/video" element={<LiveVideoScreen />} />
        <Route path="/download" element={<DownloadScreen />} />
      </Routes>
    </Router>
  );
}

// Home Screen
function Home() {
  return (
    <div style={styles.container}>
      <h1 style={styles.title}>UBIT</h1>
      <h2 style={styles.subtitle}>Real-Time Face Attendance System</h2>
      <Link to="/major">
        <button style={styles.button}>Start</button>
      </Link>
    </div>
  );
}

// Select Major Screen
function MajorScreen() {
  const [majors] = useState(["CS", "SE"]); // Example majors

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Select Major</h1>
      {majors.map((major) => (
        <Link to="/section" key={major}>
          <button style={styles.button} onClick={() => localStorage.setItem("selectedMajor", major)}>
            {major}
          </button>
        </Link>
      ))}
      <Link to="/">
        <button style={{ ...styles.button, backgroundColor: "#FF4500" }}>Back</button>
      </Link>
    </div>
  );
}

// Select Section Screen
function SectionScreen() {
  const [sections] = useState(["A", "B"]); // Example sections

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Select Section</h1>
      {sections.map((section) => (
        <Link to="/course" key={section}>
          <button style={styles.button} onClick={() => localStorage.setItem("selectedSection", section)}>
            {section}
          </button>
        </Link>
      ))}
      <Link to="/major">
        <button style={{ ...styles.button, backgroundColor: "#FF4500" }}>Back</button>
      </Link>
    </div>
  );
}

// Select Course Screen
function CourseScreen() {
  const [courses, setCourses] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState("Select Course");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCourses = async () => {
      const major = localStorage.getItem("selectedMajor");
      const section = localStorage.getItem("selectedSection");
      
      if (!major || !section) {
        setError("Major or section not selected");
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await axios.get(`/get_courses/${major}/${section}`);
        console.log("API Response:", response);  // Debug log
        
        if (Array.isArray(response.data)) {
          setCourses(response.data);
        } else {
          console.error("Unexpected response format:", response.data);
          setError("Invalid data format received");
        }
      } catch (error) {
        console.error("Error fetching courses:", error);
        setError(error.message || "Failed to fetch courses");
      } finally {
        setLoading(false);
      }
    };

    fetchCourses();
  }, []);

  if (loading) return <div style={styles.container}>Loading courses...</div>;
  if (error) return <div style={styles.container}>Error: {error}</div>;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Select Course</h1>
      <div style={styles.dropdown}>
        <button
          style={styles.dropdownButton}
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        >
          {selectedCourse} ▼
        </button>
        {isDropdownOpen && (
          <div style={styles.dropdownContent}>
            {courses.length > 0 ? (
              courses.map((course) => (
                <div
                  key={course}
                  style={styles.dropdownItem}
                  onClick={() => {
                    setSelectedCourse(course);
                    localStorage.setItem("selectedCourse", course);
                    setIsDropdownOpen(false);
                  }}
                >
                  {course}
                </div>
              ))
            ) : (
              <div style={styles.dropdownItem}>No Courses Available</div>
            )}
          </div>
        )}
      </div>
      <Link to="/video">
        <button 
          style={{ ...styles.button, backgroundColor: "#008000" }}
          disabled={selectedCourse === "Select Course"}
        >
          Take Attendance
        </button>
      </Link>
      <Link to="/section">
        <button style={{ ...styles.button, backgroundColor: "#FF4500" }}>Back</button>
      </Link>
    </div>
  );
}

// Live Video Screen
function LiveVideoScreen() {
  const videoRef = useRef(null);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [isProcessing, setIsProcessing] = useState(true);
  const processingIntervalRef = useRef(null);
  const [detectedStudents, setDetectedStudents] = useState(new Set());

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        // Start processing frames immediately
        startProcessing();
      } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Failed to access camera. Please check camera permissions and ensure the site is served over HTTPS.");
      }
    };

    // Start attendance in backend
    const startAttendance = async () => {
      const major = localStorage.getItem("selectedMajor");
      const section = localStorage.getItem("selectedSection");
      const course = localStorage.getItem("selectedCourse");

      try {
        await axios.get(`/start_attendance/${major}/${section}/${course}`);
      } catch (error) {
        console.error("Error starting attendance:", error);
        alert("Failed to start attendance session.");
      }
    };

    startCamera();
    startAttendance();

    // Cleanup function
    return () => {
      stopProcessing();
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  const processFrame = async () => {
    if (!videoRef.current || !isProcessing) return;

    const major = localStorage.getItem("selectedMajor");
    const section = localStorage.getItem("selectedSection");
    const course = localStorage.getItem("selectedCourse");

    if (!major || !section || !course) {
      alert("Missing session information. Please start over.");
      return;
    }

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frameData = canvas.toDataURL("image/jpeg").split(",")[1];

    try {
      const response = await axios.post("/process_frame", {
        frame: frameData,
        major,
        section,
        course,
      });

      setProcessedFrame(response.data.processedFrame);
      
      // Update detected students set
      if (response.data.detectedStudents) {
        setDetectedStudents(prevStudents => {
          const newStudents = new Set(prevStudents);
          response.data.detectedStudents.forEach(student => newStudents.add(student));
          return newStudents;
        });
      }
    } catch (error) {
      console.error("Error processing frame:", error);
      setIsProcessing(false);
    }
  };

  const startProcessing = () => {
    setIsProcessing(true);
    // Process frames every 1 second
    processingIntervalRef.current = setInterval(processFrame, 1000);
  };

  const stopProcessing = () => {
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    setIsProcessing(false);
  };

  const stopAttendance = async () => {
    const major = localStorage.getItem("selectedMajor");
    const section = localStorage.getItem("selectedSection");
    const course = localStorage.getItem("selectedCourse");

    if (!major || !section || !course) {
      alert("Missing session information. Please start over.");
      return;
    }

    try {
      stopProcessing();
      await axios.get(`/stop_attendance/${major}/${section}/${course}`);
      alert("Attendance stopped successfully!");
      window.location.href = "/download";
    } catch (error) {
      console.error("Error stopping attendance:", error);
      alert("Error stopping attendance. Please try again.");
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Live Attendance</h1>
      <div style={styles.videoContainer}>
        <video 
          ref={videoRef} 
          autoPlay 
          style={styles.video} 
          className={isProcessing ? styles.processingVideo : ''}
        />
        {processedFrame && (
          <img
            src={`data:image/jpeg;base64,${processedFrame}`}
            alt="Processed Frame"
            style={styles.processedVideo}
          />
        )}
      </div>
      <div style={styles.controlsContainer}>
        <div style={styles.status}>
          Status: {isProcessing ? 'Recording Attendance' : 'Stopped'}
        </div>
        <div style={styles.status}>
          Students Detected: {detectedStudents.size}
        </div>
        <button
          style={{ ...styles.button, backgroundColor: "#FF0000" }}
          onClick={stopAttendance}
        >
          Stop Attendance
        </button>
        <Link to="/course">
          <button style={{ ...styles.button, backgroundColor: "#FF4500" }}>Back</button>
        </Link>
      </div>
    </div>
  );
}

// Download Attendance Screen
function DownloadScreen() {
  const downloadExcel = async () => {
    const major = localStorage.getItem("selectedMajor");
    const section = localStorage.getItem("selectedSection");
    const course = localStorage.getItem("selectedCourse");

    if (!major || !section || !course) {
      alert("Please select major, section, and course first.");
      return;
    }

    try {
      const response = await axios.get(`/download_excel/${major}/${section}/${course}`, {
        responseType: "blob",
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "attendance.xlsx");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Error downloading Excel:", error);
      alert("Failed to download attendance record. Please try again.");
    }
  };

  const handleHomeClick = () => {
    // Clear all stored selections
    localStorage.removeItem("selectedMajor");
    localStorage.removeItem("selectedSection");
    localStorage.removeItem("selectedCourse");
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Download Attendance Record</h1>
      <button style={styles.button} onClick={downloadExcel}>
        Download Excel
      </button>
      <div style={styles.buttonContainer}>
        <Link to="/video">
          <button style={{ ...styles.button, backgroundColor: "#FF4500" }}>Back</button>
        </Link>
        <Link to="/">
          <button 
            style={{ ...styles.button, backgroundColor: "#2196F3" }}
            onClick={handleHomeClick}
          >
            Go to Home
          </button>
        </Link>
      </div>
    </div>
  );
}

// CSS Styles
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    backgroundColor: "#f5f5f5",
    padding: "20px",
    boxSizing: "border-box",
  },
  title: {
    fontSize: "2rem",
    marginBottom: "10px",
    textAlign: "center",
  },
  subtitle: {
    fontSize: "1.5rem",
    marginBottom: "20px",
    textAlign: "center",
  },
  button: {
    padding: "10px 20px",
    margin: "6px",
    fontSize: "1rem",
    backgroundColor: "#4CAF50",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    width: "100%",
    maxWidth: "200px",
  },
  buttonContainer: {
    display: "flex",
    gap: "10px",
    marginTop: "10px",
    flexWrap: "wrap",
    justifyContent: "center",
  },
  dropdown: {
    position: "relative",
    display: "inline-block",
    marginBottom: "20px",
    width: "100%",
    maxWidth: "200px",
  },
  dropdownButton: {
    padding: "10px 20px",
    fontSize: "1rem",
    backgroundColor: "#4CAF50",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    width: "100%",
  },
  dropdownContent: {
    position: "absolute",
    backgroundColor: "#f9f9f9",
    minWidth: "160px",
    boxShadow: "0px 8px 16px 0px rgba(0,0,0,0.2)",
    zIndex: 1,
    width: "100%",
  },
  dropdownItem: {
    padding: "12px 16px",
    textDecoration: "none",
    display: "block",
    color: "black",
    cursor: "pointer",
  },
  videoContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    marginBottom: '20px',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  video: {
    width: '100%',
    maxWidth: '640px',
    height: 'auto',
    backgroundColor: '#000',
    objectFit: 'cover',
  },
  processedVideo: {
    width: '100%',
    maxWidth: '640px',
    height: 'auto',
    backgroundColor: '#000',
    border: '2px solid #4CAF50',
  },
  controlsContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '10px',
    width: '100%',
  },
  status: {
    fontSize: '1.2rem',
    fontWeight: 'bold',
    color: '#333',
    marginBottom: '10px',
    textAlign: 'center',
  },
  processingVideo: {
    border: '2px solid #f00',
    animation: 'pulse 2s infinite',
  },
  '@media (max-width: 768px)': {
    title: {
      fontSize: '1.5rem',
    },
    subtitle: {
      fontSize: '1.2rem',
    },
    button: {
      fontSize: '0.9rem',
    },
    dropdownButton: {
      fontSize: '0.9rem',
    },
    status: {
      fontSize: '1rem',
    },
  },
};

export default App;
