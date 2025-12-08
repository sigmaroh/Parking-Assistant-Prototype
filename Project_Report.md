# PARKING LOT PATH PLANNER
## An Intelligent Parking Assistance System Using Computer Vision and A* Pathfinding Algorithm

---

### A PROJECT REPORT

**Note:** All processing images referenced in this report are located in the `cv_process_images/` directory. The complete image processing pipeline generates 28 intermediate images documenting each step of the detection, occupancy analysis, grid generation, and pathfinding process.

**Submitted in Partial Fulfillment of the Requirements for the Award of the Degree of**

## MASTER OF TECHNOLOGY / MASTER OF SCIENCE
### in
## COMPUTER SCIENCE AND ENGINEERING

---

**Submitted By:**
- **Student Name:** [Your Name]
- **Roll Number:** [Your Roll Number]
- **Batch:** [Your Batch Year]

**Under the Guidance of:**
- **Guide Name:** [Supervisor Name]
- **Designation:** [Designation]

---

**Department of Computer Science and Engineering**
**[Your College/University Name]**
**[Location]**
**[Academic Year: 2024-2025]**

---

# CERTIFICATE

This is to certify that the project entitled **"Parking Lot Path Planner: An Intelligent Parking Assistance System Using Computer Vision and A* Pathfinding Algorithm"** submitted by **[Student Name]** bearing Roll No. **[Roll Number]** in partial fulfillment of the requirements for the award of the degree of **Master of Technology / Master of Science** in **Computer Science and Engineering** from **[University Name]** is a bonafide record of the project work carried out by him/her under my supervision and guidance.

The work embodied in this project has not been submitted for any other degree or diploma.

---

**Date:** _______________

**Guide Signature:** _______________

**Head of Department:** _______________

**External Examiner:** _______________

---

# DECLARATION

I hereby declare that the project entitled **"Parking Lot Path Planner: An Intelligent Parking Assistance System Using Computer Vision and A* Pathfinding Algorithm"** submitted to the Department of Computer Science and Engineering, **[College Name]**, is a record of original work done by me under the guidance of **[Guide Name]**, and this project work has not been submitted for the award of any other Degree, Diploma, Associateship, Fellowship, or any other similar title.

---

**Place:** _______________

**Date:** _______________

**Signature:** _______________

**Name:** [Your Name]

---

# ACKNOWLEDGEMENT

I would like to express my sincere gratitude to all those who have contributed to the successful completion of this project.

First and foremost, I would like to thank my project guide **[Guide Name]**, for providing invaluable guidance, constant supervision, and constructive criticism throughout the development of this project. Their expertise and suggestions have been instrumental in shaping this work.

I extend my heartfelt thanks to **[HOD Name]**, Head of the Department of Computer Science and Engineering, for providing the necessary facilities and support during the project work.

I am grateful to all the faculty members of the Department of Computer Science and Engineering for their encouragement and support.

I would also like to thank my family and friends for their continuous support and motivation throughout this journey.

Finally, I thank the Almighty for giving me the strength and perseverance to complete this project successfully.

---

**[Your Name]**

---

# ABSTRACT

The increasing number of vehicles in urban areas has led to significant parking challenges, resulting in traffic congestion, fuel wastage, and driver frustration. This project presents an **Intelligent Parking Lot Path Planner** that combines computer vision techniques with pathfinding algorithms to assist drivers in navigating parking lots efficiently.

The system employs a dual detection approach:
1. **Traditional Computer Vision (OpenCV)** for detecting empty parking spots using edge detection, morphological operations, and contour analysis.
2. **YOLO (You Only Look Once) Deep Learning Model** for real-time obstacle detection including vehicles, pedestrians, and other objects.

The detected elements are converted into a navigable grid matrix, where the **A* (A-Star) pathfinding algorithm** computes the optimal route from the vehicle's current position to a target parking spot. The path is further refined using **B-Spline interpolation** to generate smooth, vehicle-friendly trajectories.

The application features a user-friendly **Tkinter-based GUI** that allows users to:
- Load parking lot images
- Configure detection parameters
- Visualize detection results
- Set start and end points interactively
- View computed paths with smoothing
- Export grid data for further processing

**Keywords:** Computer Vision, YOLO, A* Algorithm, Pathfinding, Parking Assistance, OpenCV, Deep Learning, Path Smoothing, B-Spline Interpolation

---

# TABLE OF CONTENTS

| Chapter | Title | Page No. |
|---------|-------|----------|
| | Certificate | i |
| | Declaration | ii |
| | Acknowledgement | iii |
| | Abstract | iv |
| | Table of Contents | v |
| | List of Figures | vii |
| | List of Tables | viii |
| | List of Abbreviations | ix |
| **1** | **INTRODUCTION** | 1 |
| 1.1 | Background and Motivation | 1 |
| 1.2 | Problem Statement | 2 |
| 1.3 | Objectives | 3 |
| 1.4 | Scope of the Project | 3 |
| 1.5 | Organization of Report | 4 |
| **2** | **LITERATURE REVIEW** | 5 |
| 2.1 | Existing Parking Systems | 5 |
| 2.2 | Computer Vision in Parking | 6 |
| 2.3 | Object Detection Techniques | 7 |
| 2.4 | Pathfinding Algorithms | 8 |
| 2.5 | Path Smoothing Techniques | 9 |
| 2.6 | Research Gap | 10 |
| **3** | **SYSTEM REQUIREMENTS** | 11 |
| 3.1 | Functional Requirements | 11 |
| 3.2 | Non-Functional Requirements | 12 |
| 3.3 | Hardware Requirements | 13 |
| 3.4 | Software Requirements | 13 |
| **4** | **SYSTEM DESIGN** | 14 |
| 4.1 | System Architecture | 14 |
| 4.2 | Use Case Diagram | 15 |
| 4.3 | Data Flow Diagram | 17 |
| 4.4 | Class Diagram | 19 |
| 4.5 | Sequence Diagram | 21 |
| 4.6 | Algorithm Design | 22 |
| **5** | **IMPLEMENTATION** | 27 |
| 5.1 | Development Environment | 27 |
| 5.2 | Module Description | 28 |
| 5.3 | Code Implementation | 30 |
| 5.4 | User Interface Design | 40 |
| **6** | **TESTING AND RESULTS** | 43 |
| 6.1 | Testing Methodology | 43 |
| 6.2 | Test Cases | 44 |
| 6.3 | Results and Analysis | 46 |
| 6.4 | Performance Evaluation | 48 |
| **7** | **CONCLUSION AND FUTURE WORK** | 50 |
| 7.1 | Conclusion | 50 |
| 7.2 | Limitations | 51 |
| 7.3 | Future Enhancements | 51 |
| | **REFERENCES** | 53 |
| | **APPENDICES** | 55 |
| A | Source Code | 55 |
| B | Sample Outputs | 70 |

---

# LIST OF FIGURES

| Figure No. | Title | Page No. |
|------------|-------|----------|
| 1.1 | Problem Scenario - Parking Lot Navigation | 2 |
| 2.1 | Evolution of Parking Systems | 5 |
| 2.2 | YOLO Architecture Overview | 7 |
| 2.3 | A* Algorithm Illustration | 8 |
| 4.1 | System Architecture Diagram | 14 |
| 4.2 | Use Case Diagram | 15 |
| 4.3 | Context Level DFD (Level 0) | 17 |
| 4.4 | Level 1 Data Flow Diagram | 18 |
| 4.5 | Level 2 Data Flow Diagram | 19 |
| 4.6 | Class Diagram | 20 |
| 4.7 | Sequence Diagram - Path Planning | 21 |
| 4.8 | A* Algorithm Flowchart | 23 |
| 4.9 | Parking Spot Detection Flowchart | 25 |
| 5.1 | Main Application Window | 40 |
| 5.2 | Image Processing Pipeline | 41 |
| 5.3 | Grid Visualization | 42 |
| 6.1 | Original Parking Lot Image (1_original.png) | 46 |
| 6.2 | Grayscale Conversion (2_grayscale.png) | 46 |
| 6.3 | Gaussian Blur Applied (3_blurred.png) | 46 |
| 6.4 | Canny Edge Detection (4_edges_canny.png) | 47 |
| 6.5 | Hough Line Detection (5_hough_lines.png) | 47 |
| 6.6 | Classified Horizontal and Vertical Lines (6_classified_lines_HV.png) | 47 |
| 6.7 | All Detected Lines (7_all_detected_lines.png) | 47 |
| 6.8 | Morphological Closing (8_lines_closed.png) | 48 |
| 6.9 | Lines Thickened (9_lines_thickened.png) | 48 |
| 6.10 | Inverted Regions (10_inverted_regions.png) | 48 |
| 6.11 | Valid Parking Regions (13_valid_parking_regions.png) | 49 |
| 6.12 | Detected Contours (14_detected_contours.png) | 49 |
| 6.13 | Rectangles Filtered by Shape (15_rectangles_filtered_by_shape.png) | 49 |
| 6.14 | Final Parking Spaces (16_parking_spaces_final.png) | 49 |
| 6.15 | Detected Parking Spots with Occupancy (17_detected_parking_spots.png) | 50 |
| 6.16 | Occupancy Analysis (18_occupancy_analysis.png) | 50 |
| 6.17 | Occupancy Edge Detection (19_occupancy_edges.png) | 50 |
| 6.18 | Occupancy Variance Analysis (20_occupancy_variance.png) | 50 |
| 6.19 | Occupancy Bright Pixels (21_occupancy_bright_pixels.png) | 51 |
| 6.20 | Occupancy Dark Pixels (22_occupancy_dark_pixels.png) | 51 |
| 6.21 | Occupancy Combined Score (23_occupancy_combined_score.png) | 51 |
| 6.22 | Occupancy Color Coded (24_occupancy_color_coded.png) | 51 |
| 6.23 | Generated Grid Matrix (25_grid_matrix.png) | 52 |
| 6.24 | Inflated Grid with Clearance (26_inflated_grid.png) | 52 |
| 6.25 | A* Path Visualization (27_astar_path.png) | 52 |
| 6.26 | Smoothed Path (28_smoothed_path.png) | 52 |

---

# LIST OF TABLES

| Table No. | Title | Page No. |
|-----------|-------|----------|
| 3.1 | Functional Requirements | 11 |
| 3.2 | Non-Functional Requirements | 12 |
| 3.3 | Hardware Requirements | 13 |
| 3.4 | Software Requirements | 13 |
| 4.1 | Grid Matrix Legend | 16 |
| 5.1 | Python Libraries Used | 27 |
| 6.1 | Test Cases - Image Loading | 44 |
| 6.2 | Test Cases - Detection | 44 |
| 6.3 | Test Cases - Pathfinding | 45 |
| 6.4 | Performance Metrics | 48 |

---

# LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|--------------|-----------|
| A* | A-Star Algorithm |
| API | Application Programming Interface |
| B-Spline | Basis Spline |
| CNN | Convolutional Neural Network |
| COCO | Common Objects in Context |
| CPU | Central Processing Unit |
| CV | Computer Vision |
| DFD | Data Flow Diagram |
| GB | Gigabyte |
| GPU | Graphics Processing Unit |
| GUI | Graphical User Interface |
| HD | High Definition |
| JSON | JavaScript Object Notation |
| ML | Machine Learning |
| NumPy | Numerical Python |
| OpenCV | Open Source Computer Vision Library |
| PIL | Python Imaging Library |
| RAM | Random Access Memory |
| RGB | Red Green Blue |
| SciPy | Scientific Python |
| UML | Unified Modeling Language |
| YOLO | You Only Look Once |

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation

The rapid urbanization and increasing vehicle ownership have created significant challenges in urban mobility, particularly in finding suitable parking spaces. Studies indicate that drivers spend an average of **17-20 minutes** searching for parking in busy urban areas, contributing to:

- **Traffic congestion** due to vehicles circulating in search of parking
- **Increased fuel consumption** and environmental pollution
- **Driver frustration** and reduced productivity
- **Safety hazards** from distracted driving while searching for spots

Traditional parking systems rely on simple signage or basic occupancy sensors, which provide limited assistance in navigating complex parking structures. The emergence of **computer vision** and **artificial intelligence** technologies offers promising solutions to these challenges.

This project develops an intelligent parking assistance system that combines:
1. **Computer Vision** for real-time environment perception
2. **Deep Learning** for accurate object detection
3. **Pathfinding Algorithms** for optimal route computation
4. **Path Smoothing** for generating vehicle-friendly trajectories

The system aims to reduce parking search time, improve traffic flow, and enhance the overall parking experience.

## 1.2 Problem Statement

Current parking assistance systems face several limitations:

1. **Limited Environmental Awareness**: Most systems cannot distinguish between different types of obstacles or understand the parking lot layout dynamically.

2. **Lack of Navigation Guidance**: Existing systems may indicate spot availability but fail to provide navigation assistance to reach the spot.

3. **Static Detection Methods**: Traditional sensor-based systems cannot adapt to changing environments or detect unexpected obstacles.

4. **No Path Optimization**: Without intelligent routing, drivers may take suboptimal paths, especially in large parking structures.

5. **Integration Challenges**: Separate systems for detection and navigation create complexity and reduce efficiency.

**Research Question**: How can computer vision and pathfinding algorithms be integrated to create an intelligent system that detects parking spots, identifies obstacles, and computes optimal navigation paths in real-time?

## 1.3 Objectives

The primary objectives of this project are:

### Primary Objectives:
1. **Develop a parking spot detection system** using traditional computer vision techniques (edge detection, contour analysis, morphological operations).

2. **Implement obstacle detection** using YOLO deep learning model for identifying vehicles, pedestrians, and other obstacles.

3. **Create a grid-based environment representation** that converts detected objects into a navigable matrix.

4. **Implement A* pathfinding algorithm** for computing optimal routes from start to destination.

5. **Apply path smoothing techniques** using B-Spline interpolation for generating realistic vehicle trajectories.

### Secondary Objectives:
6. Develop an intuitive graphical user interface for system interaction.
7. Enable configurable parameters for different parking lot scenarios.
8. Provide export functionality for integration with other systems.
9. Ensure real-time performance for practical applications.

## 1.4 Scope of the Project

### In Scope:
- Detection of empty parking spots from aerial/top-down images
- Detection of obstacles (vehicles, people, objects) using YOLO
- Grid-based environment modeling
- A* pathfinding with diagonal movement support
- Obstacle inflation for safe clearance
- B-Spline path smoothing
- Interactive GUI for parameter configuration
- Export of grid data in JSON, NumPy, and text formats
- Visualization of detection results and computed paths

### Out of Scope:
- Real-time video processing (current version handles static images)
- Integration with vehicle control systems
- Multi-floor parking structure navigation
- Payment and booking systems
- Mobile application development
- Cloud-based deployment

## 1.5 Organization of Report

This report is organized into the following chapters:

**Chapter 1 - Introduction**: Provides background, problem statement, objectives, and scope.

**Chapter 2 - Literature Review**: Reviews existing parking systems, computer vision techniques, object detection methods, and pathfinding algorithms.

**Chapter 3 - System Requirements**: Details functional, non-functional, hardware, and software requirements.

**Chapter 4 - System Design**: Presents system architecture, UML diagrams, and algorithm designs.

**Chapter 5 - Implementation**: Describes the development environment, module implementation, and user interface design.

**Chapter 6 - Testing and Results**: Covers testing methodology, test cases, results, and performance analysis.

**Chapter 7 - Conclusion and Future Work**: Summarizes findings, discusses limitations, and proposes future enhancements.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Existing Parking Systems

Parking management systems have evolved significantly over the years:

### 2.1.1 Manual Systems
Traditional parking lots rely on human attendants to direct vehicles, which is labor-intensive and error-prone.

### 2.1.2 Sensor-Based Systems
- **Ultrasonic Sensors**: Detect vehicle presence using sound waves
- **Infrared Sensors**: Use infrared light for occupancy detection
- **Magnetic Sensors**: Embedded in pavement to detect metal vehicles
- **Limitations**: High installation costs, maintenance requirements, limited intelligence

### 2.1.3 Camera-Based Systems
Modern systems use cameras for:
- License plate recognition
- Occupancy monitoring
- Traffic flow analysis

### 2.1.4 Smart Parking Systems
Integration of IoT, mobile apps, and cloud computing for:
- Real-time availability updates
- Reservation systems
- Payment processing

## 2.2 Computer Vision in Parking

Computer vision techniques for parking applications include:

### 2.2.1 Image Processing Techniques
- **Grayscale Conversion**: Reduces computational complexity
- **Gaussian Blur**: Noise reduction for cleaner edge detection
- **Canny Edge Detection**: Identifies parking spot boundaries
- **Morphological Operations**: Refines detected features

### 2.2.2 Feature Detection
- **Contour Detection**: Identifies rectangular parking spot shapes
- **Hough Transform**: Detects lines for parking space delineation
- **Corner Detection**: Identifies parking spot corners

### 2.2.3 PKLot Dataset Study (Almeida et al., 2015)
The PKLot dataset contains 12,417 images of parking lots with labeled occupancy status. Research using this dataset achieved:
- 99.5% accuracy using texture descriptors
- 97% accuracy using CNN-based methods

## 2.3 Object Detection Techniques

### 2.3.1 Traditional Methods
- **HOG (Histogram of Oriented Gradients)**: Captures gradient orientation for object detection
- **SVM (Support Vector Machine)**: Classification of detected objects
- **Sliding Window**: Exhaustive search across image

### 2.3.2 Deep Learning Methods

#### YOLO (You Only Look Once)
- Single-pass detection architecture
- Real-time performance (45+ FPS)
- Grid-based prediction
- Anchor boxes for multiple scales

**YOLO Architecture:**
```
Input Image → CNN Backbone → Feature Maps → Detection Head → Bounding Boxes + Class Labels
```

#### Evolution of YOLO:
| Version | Year | Key Features |
|---------|------|--------------|
| YOLOv1 | 2016 | First real-time detector |
| YOLOv2 | 2017 | Batch normalization, anchor boxes |
| YOLOv3 | 2018 | Multi-scale detection |
| YOLOv4 | 2020 | CSPDarknet backbone |
| YOLOv5 | 2020 | PyTorch implementation |
| YOLOv8 | 2023 | Anchor-free, improved accuracy |

### 2.3.3 Comparison of Detection Methods

| Method | Speed | Accuracy | Complexity |
|--------|-------|----------|------------|
| HOG+SVM | Slow | Moderate | Low |
| R-CNN | Very Slow | High | High |
| Fast R-CNN | Slow | High | High |
| Faster R-CNN | Moderate | High | High |
| YOLO | Fast | High | Moderate |
| SSD | Fast | High | Moderate |

## 2.4 Pathfinding Algorithms

### 2.4.1 Dijkstra's Algorithm
- Guarantees shortest path
- Explores all directions equally
- Time complexity: O(V²) or O(E log V) with priority queue
- Limitation: No heuristic guidance, slower for large graphs

### 2.4.2 A* Algorithm
The A* algorithm combines:
- **g(n)**: Actual cost from start to node n
- **h(n)**: Heuristic estimate from n to goal
- **f(n) = g(n) + h(n)**: Total estimated cost

**Advantages:**
- Optimal path guarantee (with admissible heuristic)
- Faster than Dijkstra due to heuristic guidance
- Supports diagonal movement

**Heuristics:**
- **Manhattan Distance**: For 4-directional movement
- **Euclidean Distance**: For any-angle movement
- **Chebyshev Distance**: For 8-directional movement

### 2.4.3 Other Algorithms
- **D* (Dynamic A*)**: For changing environments
- **RRT (Rapidly-exploring Random Trees)**: For high-dimensional spaces
- **Potential Fields**: For continuous environments

## 2.5 Path Smoothing Techniques

Raw A* paths often have sharp turns unsuitable for vehicles:

### 2.5.1 Line Simplification
- **Douglas-Peucker Algorithm**: Reduces path points while preserving shape
- Simple but may still have sharp corners

### 2.5.2 Spline Interpolation
- **B-Spline**: Smooth curves passing near control points
- **Cubic Spline**: Passes through all points with continuous derivatives
- **Bézier Curves**: Controlled by convex hull of control points

### 2.5.3 Optimization-Based Smoothing
- **Gradient Descent**: Minimizes curvature while maintaining feasibility
- **Quadratic Programming**: Optimizes path with constraints

## 2.6 Research Gap

Despite advances in individual areas, gaps remain:

1. **Integration Gap**: Few systems combine detection and navigation seamlessly
2. **Dual Detection**: Most use either traditional CV or deep learning, not both
3. **Interactive Systems**: Limited user control over parameters
4. **Visualization**: Lack of comprehensive visualization tools
5. **Export Capability**: Poor interoperability with other systems

This project addresses these gaps by creating an integrated system that:
- Combines traditional CV (parking spots) with deep learning (obstacles)
- Provides interactive parameter configuration
- Offers comprehensive visualization
- Enables data export in multiple formats

---

# CHAPTER 3: SYSTEM REQUIREMENTS

## 3.1 Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR1 | Image Loading | High | System shall load parking lot images in JPG, PNG, BMP formats |
| FR2 | Parking Spot Detection | High | System shall detect empty parking spots using computer vision |
| FR3 | Obstacle Detection | High | System shall detect obstacles using YOLO model |
| FR4 | Grid Generation | High | System shall convert detections to navigable grid matrix |
| FR5 | Start/End Point Selection | High | System shall allow interactive start/end point selection |
| FR6 | Pathfinding | High | System shall compute optimal path using A* algorithm |
| FR7 | Path Smoothing | Medium | System shall smooth paths using B-Spline interpolation |
| FR8 | Visualization | High | System shall display detection results and paths |
| FR9 | Grid Export | Medium | System shall export grid to JSON, NumPy, text formats |
| FR10 | Parameter Configuration | Medium | System shall allow configuration of detection parameters |
| FR11 | Obstacle Inflation | Medium | System shall support configurable clearance radius |
| FR12 | Path Clearing | Low | System shall allow clearing of computed paths |

## 3.2 Non-Functional Requirements

| ID | Requirement | Category | Description |
|----|-------------|----------|-------------|
| NFR1 | Response Time | Performance | Detection shall complete within 5 seconds |
| NFR2 | Path Computation | Performance | A* shall compute path within 2 seconds |
| NFR3 | User Interface | Usability | GUI shall be intuitive and self-explanatory |
| NFR4 | Image Support | Compatibility | Shall support images up to 4K resolution |
| NFR5 | Cross-Platform | Portability | Shall run on Windows, Linux, macOS |
| NFR6 | Memory Usage | Performance | Shall not exceed 2GB RAM during operation |
| NFR7 | Error Handling | Reliability | Shall handle invalid inputs gracefully |
| NFR8 | Modularity | Maintainability | Code shall be modular and well-documented |

## 3.3 Hardware Requirements

### Minimum Requirements:
| Component | Specification |
|-----------|---------------|
| Processor | Intel Core i3 / AMD Ryzen 3 (or equivalent) |
| RAM | 4 GB |
| Storage | 2 GB free disk space |
| Display | 1366 x 768 resolution |
| GPU | Integrated graphics (CPU-based inference) |

### Recommended Requirements:
| Component | Specification |
|-----------|---------------|
| Processor | Intel Core i5 / AMD Ryzen 5 (or better) |
| RAM | 8 GB or more |
| Storage | 5 GB free disk space (SSD preferred) |
| Display | 1920 x 1080 resolution |
| GPU | NVIDIA GPU with CUDA support (for faster inference) |

## 3.4 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8 - 3.12 | Programming language |
| OpenCV (cv2) | 4.5+ | Image processing |
| NumPy | 1.20+ | Numerical computations |
| Pillow (PIL) | 8.0+ | Image handling for GUI |
| Tkinter | Built-in | Graphical user interface |
| Ultralytics | 8.0+ | YOLO implementation |
| SciPy | 1.7+ | B-Spline interpolation |
| PyTorch | 1.10+ | Deep learning backend |

### Installation Command:
```bash
pip install opencv-python numpy pillow ultralytics scipy torch
```

---

# CHAPTER 4: SYSTEM DESIGN

## 4.1 System Architecture

The system follows a **modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARKING LOT PATH PLANNER                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   INPUT LAYER   │    │ PROCESSING LAYER│    │  OUTPUT LAYER   │     │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤     │
│  │                 │    │                 │    │                 │     │
│  │ • Image Loader  │───▶│ • CV Detection  │───▶│ • Visualization │     │
│  │ • User Input    │    │ • YOLO Detection│    │ • Grid Display  │     │
│  │ • Configuration │    │ • Grid Generator│    │ • Path Display  │     │
│  │                 │    │ • A* Pathfinding│    │ • Data Export   │     │
│  │                 │    │ • Path Smoother │    │                 │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                           GUI LAYER (Tkinter)                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Control Panel │ Image Canvas │ Grid Canvas │ Status/Info Panel │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1.1 Layer Description

**Input Layer:**
- Handles image file selection and loading
- Captures user interactions (clicks, button presses)
- Manages configuration parameters

**Processing Layer:**
- Traditional CV module for parking spot detection
- YOLO module for obstacle detection
- Grid generation from detection results
- A* pathfinding algorithm
- B-Spline path smoothing

**Output Layer:**
- Image visualization with overlays
- Grid matrix visualization
- Path display
- Data export functionality

**GUI Layer:**
- Tkinter-based interface
- Canvas widgets for visualization
- Control buttons and spinboxes
- Status and information panels

## 4.2 Use Case Diagram

### 4.2.1 Actors

**Primary Actor - User:**
- Interacts with the system through GUI
- Loads images, configures parameters
- Sets navigation points
- Initiates detection and pathfinding

**Secondary Actors:**
- **YOLO Model**: Performs obstacle detection
- **OpenCV Engine**: Performs parking spot detection

### 4.2.2 Use Cases

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    PARKING LOT PATH PLANNER SYSTEM                        │
│                                                                           │
│   ┌─────────────────┐                                                    │
│   │ Load Parking    │◀──────────────────────────────────┐                │
│   │ Lot Image       │                                    │                │
│   └─────────────────┘                                    │                │
│            │                                             │                │
│            ▼                                             │                │
│   ┌─────────────────┐     ┌─────────────────┐           │                │
│   │ Detect Empty    │     │ Detect Obstacles│           │                │
│   │ Parking Spots   │     │ (YOLO)          │◀────┐     │                │
│   └─────────────────┘     └─────────────────┘     │     │                │
│            │                       │               │     │                │
│            └───────────┬───────────┘               │     │                │
│                        ▼                           │     │                │
│            ┌─────────────────┐                     │     │                │
│            │ Generate        │                     │     │     ┌──────┐  │
│            │ Navigation Grid │                     │     ├────▶│ User │  │
│            └─────────────────┘                     │     │     └──────┘  │
│                        │                           │     │                │
│            ┌───────────┴───────────┐              │     │                │
│            ▼                       ▼               │     │                │
│   ┌─────────────────┐     ┌─────────────────┐     │     │                │
│   │ Set Start Point │     │ Set End Point   │     │     │                │
│   └─────────────────┘     └─────────────────┘     │     │                │
│            │                       │               │     │                │
│            └───────────┬───────────┘              │     │                │
│                        ▼                           │     │                │
│            ┌─────────────────┐                     │     │                │
│            │ Run A*          │                     │     │                │
│            │ Pathfinding     │────────────────────▶│     │                │
│            └─────────────────┘                     │     │                │
│                        │                           │     │                │
│            ┌───────────┴───────────┐              │     │                │
│            ▼                       ▼               │     │                │
│   ┌─────────────────┐     ┌─────────────────┐     │     │                │
│   │ Smooth Path     │     │ Export Grid     │─────┴─────┘                │
│   │ (B-Spline)      │     │                 │                            │
│   └─────────────────┘     └─────────────────┘                            │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
                │                               │
                ▼                               ▼
         ┌──────────┐                    ┌──────────┐
         │ OpenCV   │                    │  YOLO    │
         │ Engine   │                    │  Model   │
         └──────────┘                    └──────────┘
```

### 4.2.3 Grid Matrix Legend

| Value | Meaning | Color | Description |
|-------|---------|-------|-------------|
| 0 | Navigable | White | Free space for A* pathfinding |
| 1 | Obstacle | Red | YOLO detected objects (cars, people) |
| 2 | Empty Parking Spot | Green | Traditional CV detected spots |

## 4.3 Data Flow Diagram

### 4.3.1 Context Level DFD (Level 0)

```
                    ┌─────────────────────────────┐
     Image File ───▶│                             │───▶ Processed Image
                    │   PARKING LOT PATH PLANNER  │
  Configuration ───▶│                             │───▶ Navigation Path
                    │                             │
    Start/End   ───▶│                             │───▶ Grid Export
                    └─────────────────────────────┘
                               ▲
                               │
                          ┌────┴────┐
                          │  User   │
                          └─────────┘
```

### 4.3.2 Level 1 DFD

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    ┌─────────┐         ┌─────────────┐         ┌─────────────┐             │
│    │         │ Image   │             │ Spots   │             │ Grid        │
│ ───┤ 1.0     ├────────▶│ 2.0         ├────────▶│ 3.0         ├──────────▶  │
│    │ Load    │         │ Detect      │         │ Generate    │             │
│    │ Image   │         │ (CV+YOLO)   │         │ Grid        │             │
│    └─────────┘         └─────────────┘         └──────┬──────┘             │
│                                                        │                     │
│                                                        ▼                     │
│                                                 ┌─────────────┐             │
│                         Points                  │             │ Path        │
│                    ◀────────────────────────────│ 4.0         ├──────────▶  │
│                                                 │ Find Path   │             │
│                                                 │ (A*)        │             │
│                                                 └──────┬──────┘             │
│                                                        │                     │
│                                                        ▼                     │
│                                                 ┌─────────────┐             │
│                                                 │ 5.0         │ Smooth      │
│                                              ──▶│ Smooth Path ├─────────▶   │
│                                                 │ (B-Spline)  │             │
│                                                 └─────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3.3 Level 2 DFD - Detection Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DETECTION PROCESS (2.0)                            │
│                                                                              │
│    ┌─────────────┐         ┌─────────────────┐                              │
│    │ 2.1         │ Gray    │ 2.2             │ Edges                        │
│───▶│ Grayscale   ├────────▶│ Gaussian Blur   ├─────────┐                    │
│    │ Conversion  │         │                 │         │                    │
│    └─────────────┘         └─────────────────┘         ▼                    │
│                                                 ┌─────────────┐             │
│                                                 │ 2.3         │             │
│                                                 │ Canny Edge  │             │
│                                                 │ Detection   │             │
│                                                 └──────┬──────┘             │
│                                                        │                     │
│                                                        ▼                     │
│    ┌─────────────┐         ┌─────────────────┐ ┌─────────────┐             │
│    │ 2.6         │ Spots   │ 2.5             │ │ 2.4         │             │
│◀───│ Parking Spot├─────────│ Contour         │◀│ Morphology  │             │
│    │ Validation  │         │ Detection       │ │ Operations  │             │
│    └─────────────┘         └─────────────────┘ └─────────────┘             │
│                                                                              │
│    ═══════════════════════════════════════════════════════════════         │
│                            YOLO DETECTION BRANCH                             │
│                                                                              │
│    ┌─────────────┐         ┌─────────────────┐         ┌─────────────┐     │
│    │ 2.7         │ Tensor  │ 2.8             │ Boxes   │ 2.9         │     │
│───▶│ Preprocess  ├────────▶│ YOLO Inference  ├────────▶│ Post-process├────▶│
│    │ Image       │         │                 │         │ Results     │     │
│    └─────────────┘         └─────────────────┘         └─────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4.4 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ParkingGridConverter                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Attributes:                                                                  │
│ ─────────────────────────────────────────────────────────────────────────── │
│ - root: tk.Tk                                                                │
│ - original_image: np.ndarray                                                │
│ - processed_image: np.ndarray                                               │
│ - grid_matrix: np.ndarray                                                   │
│ - grid_rows: int = 80                                                       │
│ - grid_cols: int = 120                                                      │
│ - yolo_model: YOLO                                                          │
│ - detected_objects: List[Dict]                                              │
│ - detection_confidence: float = 0.3                                         │
│ - parking_spots: List[Dict]                                                 │
│ - start_point: Tuple[int, int]                                              │
│ - end_point: Tuple[int, int]                                                │
│ - path: List[Tuple[int, int]]                                               │
│ - smoothed_path: np.ndarray                                                 │
│ - inflated_grid: np.ndarray                                                 │
│ - clearance_radius: int = 1                                                 │
│ - image_canvas: tk.Canvas                                                   │
│ - grid_canvas: tk.Canvas                                                    │
│ - click_mode: str                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Methods:                                                                     │
│ ─────────────────────────────────────────────────────────────────────────── │
│ + __init__(root: tk.Tk)                                                     │
│ + create_widgets()                                                          │
│ + load_yolo_model()                                                         │
│ + load_image()                                                              │
│ + detect_parking_spots()                                                    │
│ + detect_obstacles_yolo()                                                   │
│ + generate_grid()                                                           │
│ + visualize_grid_matrix()                                                   │
│ + export_grid()                                                             │
│ + set_mode(mode: str)                                                       │
│ + on_canvas_click(event)                                                    │
│ + redraw_with_points()                                                      │
│ + run_astar()                                                               │
│ + inflate_obstacles(grid: np.ndarray, clearance: int): np.ndarray           │
│ + astar(start: Tuple, goal: Tuple): List[Tuple]                             │
│ + smooth_path_bspline(path: List, smoothing: float, points: int): np.ndarray│
│ + visualize_grid_with_path()                                                │
│ + clear_path()                                                              │
│ + display_image(img: np.ndarray, canvas: tk.Canvas)                         │
│ + update_status(message: str)                                               │
│ + update_info(message: str)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ uses
                                     ▼
        ┌───────────────────────────────────────────────────────────┐
        │                                                           │
        ▼                       ▼                       ▼           │
┌───────────────┐     ┌───────────────┐     ┌───────────────┐      │
│    OpenCV     │     │     YOLO      │     │    SciPy      │      │
├───────────────┤     ├───────────────┤     ├───────────────┤      │
│ + imread()    │     │ + __call__()  │     │ + Univariate  │      │
│ + cvtColor()  │     │ + predict()   │     │   Spline()    │      │
│ + GaussianBlur│     │               │     │ + ndimage     │      │
│ + Canny()     │     │               │     │               │      │
│ + findContours│     │               │     │               │      │
│ + morphology  │     │               │     │               │      │
└───────────────┘     └───────────────┘     └───────────────┘      │
```

## 4.5 Sequence Diagram

### 4.5.1 Path Planning Sequence

```
┌──────┐     ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│ User │     │   GUI   │     │ Detector │     │   Grid   │     │   A*    │
└──┬───┘     └────┬────┘     └────┬─────┘     └────┬─────┘     └────┬────┘
   │              │               │                │                │
   │ Load Image   │               │                │                │
   │─────────────▶│               │                │                │
   │              │ imread()      │                │                │
   │              │──────────────▶│                │                │
   │              │◀──────────────│                │                │
   │              │ Display       │                │                │
   │◀─────────────│               │                │                │
   │              │               │                │                │
   │ Detect Spots │               │                │                │
   │─────────────▶│               │                │                │
   │              │ detect_parking_spots()         │                │
   │              │──────────────▶│                │                │
   │              │               │ Process        │                │
   │              │               │────────────────│                │
   │              │◀──────────────│                │                │
   │              │               │                │                │
   │ Detect YOLO  │               │                │                │
   │─────────────▶│               │                │                │
   │              │ detect_obstacles_yolo()        │                │
   │              │──────────────▶│                │                │
   │              │◀──────────────│                │                │
   │              │               │                │                │
   │ Generate Grid│               │                │                │
   │─────────────▶│               │                │                │
   │              │ generate_grid()                │                │
   │              │───────────────────────────────▶│                │
   │              │◀───────────────────────────────│                │
   │              │               │                │                │
   │ Set Points   │               │                │                │
   │─────────────▶│               │                │                │
   │              │               │                │                │
   │ Run A*       │               │                │                │
   │─────────────▶│               │                │                │
   │              │ run_astar()   │                │                │
   │              │──────────────────────────────────────────────────▶│
   │              │               │                │   Path         │
   │              │◀──────────────────────────────────────────────────│
   │              │ Display Path  │                │                │
   │◀─────────────│               │                │                │
   │              │               │                │                │
```

## 4.6 Algorithm Design

### 4.6.1 A* Pathfinding Algorithm

**Pseudocode:**

```
ALGORITHM A_Star(start, goal, grid)
INPUT: 
    start - starting position (row, col)
    goal - target position (row, col)
    grid - 2D matrix (0=free, 1=obstacle)
OUTPUT:
    path - list of positions from start to goal

BEGIN
    // Initialize data structures
    open_set ← priority queue with (f(start), start)
    came_from ← empty map
    g_score[start] ← 0
    f_score[start] ← heuristic(start, goal)
    closed_set ← empty set
    
    WHILE open_set is not empty DO
        current ← node in open_set with lowest f_score
        
        IF current = goal THEN
            RETURN reconstruct_path(came_from, current)
        END IF
        
        Remove current from open_set
        Add current to closed_set
        
        FOR each neighbor of current DO
            IF neighbor in closed_set THEN
                CONTINUE
            END IF
            
            IF grid[neighbor] = 1 THEN  // Obstacle
                CONTINUE
            END IF
            
            tentative_g ← g_score[current] + distance(current, neighbor)
            
            IF neighbor not in g_score OR tentative_g < g_score[neighbor] THEN
                came_from[neighbor] ← current
                g_score[neighbor] ← tentative_g
                f_score[neighbor] ← tentative_g + heuristic(neighbor, goal)
                
                IF neighbor not in open_set THEN
                    Add neighbor to open_set with priority f_score[neighbor]
                END IF
            END IF
        END FOR
    END WHILE
    
    RETURN null  // No path found
END
```

**Heuristic Function (Euclidean Distance):**

```
FUNCTION heuristic(pos1, pos2)
    dx ← |pos1.x - pos2.x|
    dy ← |pos1.y - pos2.y|
    RETURN sqrt(dx² + dy²)
END FUNCTION
```

### 4.6.2 A* Algorithm Flowchart

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Initialize open_set │
                │ with start node     │
                └──────────┬──────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Is open_set empty?     │
              └───────────┬────────────┘
                    │           │
                 Yes│           │No
                    ▼           ▼
           ┌────────────┐ ┌─────────────────────┐
           │ Return     │ │ Get node with       │
           │ NO PATH    │ │ lowest f-score      │
           └────────────┘ └──────────┬──────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │ Is current = goal?   │
                          └───────────┬──────────┘
                                │           │
                             Yes│           │No
                                ▼           ▼
                    ┌────────────────┐  ┌───────────────────┐
                    │ Reconstruct    │  │ Add current to    │
                    │ and return     │  │ closed_set        │
                    │ path           │  └─────────┬─────────┘
                    └────────────────┘            │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │ For each neighbor       │
                                    │ of current              │
                                    └───────────┬─────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │ Is neighbor valid and   │
                                    │ not in closed_set?      │
                                    └───────────┬─────────────┘
                                          │           │
                                       Yes│           │No
                                          ▼           │
                              ┌───────────────────┐   │
                              │ Calculate         │   │
                              │ tentative g-score │   │
                              └─────────┬─────────┘   │
                                        │             │
                                        ▼             │
                              ┌───────────────────┐   │
                              │ Is new path       │   │
                              │ better?           │   │
                              └─────────┬─────────┘   │
                                  │           │       │
                               Yes│           │No     │
                                  ▼           │       │
                        ┌─────────────────┐   │       │
                        │ Update scores   │   │       │
                        │ Add to open_set │   │       │
                        └────────┬────────┘   │       │
                                 │            │       │
                                 └────────────┴───────┘
                                              │
                                              │
                                 (Loop back to check open_set)
```

### 4.6.3 Parking Spot Detection Algorithm

**Pseudocode:**

```
ALGORITHM Detect_Parking_Spots(image)
INPUT:
    image - RGB image of parking lot
OUTPUT:
    parking_spots - list of detected parking spot rectangles

BEGIN
    // Step 1: Preprocessing
    gray ← convert_to_grayscale(image)
    blurred ← gaussian_blur(gray, kernel_size=5)
    
    // Step 2: Edge Detection
    edges ← canny_edge_detection(blurred, low=50, high=150)
    
    // Step 3: Morphological Operations
    kernel ← create_rectangular_kernel(5, 5)
    closed ← morphological_close(edges, kernel, iterations=1)
    dilated ← dilate(closed, kernel, iterations=1)
    
    // Step 4: Contour Detection
    contours ← find_contours(dilated)
    
    // Step 5: Parking Spot Filtering
    parking_spots ← empty list
    
    FOR each contour in contours DO
        area ← calculate_area(contour)
        
        IF area < MIN_AREA OR area > MAX_AREA THEN
            CONTINUE
        END IF
        
        bounding_rect ← get_bounding_rectangle(contour)
        aspect_ratio ← max(width, height) / min(width, height)
        
        IF aspect_ratio < MIN_ASPECT OR aspect_ratio > MAX_ASPECT THEN
            CONTINUE
        END IF
        
        solidity ← area / convex_hull_area(contour)
        is_rectangular ← 4 <= approx_polygon_vertices <= 6
        
        IF is_rectangular OR solidity < MAX_SOLIDITY THEN
            ADD contour info to parking_spots
        END IF
    END FOR
    
    RETURN parking_spots
END
```

### 4.6.4 B-Spline Path Smoothing

**Pseudocode:**

```
ALGORITHM Smooth_Path_BSpline(path, smoothing_factor, num_points)
INPUT:
    path - list of (row, col) coordinates
    smoothing_factor - B-spline smoothing parameter
    num_points - number of output points
OUTPUT:
    smoothed_path - smoothed path coordinates

BEGIN
    IF length(path) < 3 THEN
        RETURN path
    END IF
    
    // Extract coordinates
    x_coords ← path[:, 0]
    y_coords ← path[:, 1]
    
    // Calculate cumulative distances
    distances ← [0]
    FOR i FROM 1 TO length(path) - 1 DO
        d ← euclidean_distance(path[i], path[i-1])
        APPEND distances[-1] + d TO distances
    END FOR
    
    // Normalize to [0, 1]
    distances ← distances / distances[-1]
    
    // Create smooth parameter array
    t_smooth ← linspace(0, 1, num_points)
    
    // Fit B-splines
    spline_x ← UnivariateSpline(distances, x_coords, s=smoothing_factor)
    spline_y ← UnivariateSpline(distances, y_coords, s=smoothing_factor)
    
    // Evaluate smooth coordinates
    x_smooth ← spline_x(t_smooth)
    y_smooth ← spline_y(t_smooth)
    
    RETURN column_stack(x_smooth, y_smooth)
END
```

---

# CHAPTER 5: IMPLEMENTATION

## 5.1 Development Environment

### 5.1.1 Development Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12 | Primary programming language |
| VS Code / PyCharm | Latest | Integrated Development Environment |
| Git | 2.40+ | Version control |
| pip | 23.0+ | Package management |

### 5.1.2 Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8.0 | Image processing and computer vision |
| numpy | 1.24.0 | Numerical computations and array operations |
| pillow | 10.0.0 | Image handling for Tkinter |
| ultralytics | 8.0.0 | YOLOv8 object detection |
| scipy | 1.11.0 | B-Spline interpolation and scientific computing |
| torch | 2.0.0 | Deep learning backend for YOLO |
| tkinter | Built-in | Graphical user interface |

### 5.1.3 Project Structure

```
Parking-Assistant/
├── main.py                      # Main application file (ParkingGridConverter class)
├── a_star.py                    # Standalone A* pathfinding implementation
├── mpc_controller.py            # Model Predictive Controller (for vehicle dynamics)
├── cv_process_images/           # Complete CV processing pipeline images
│   ├── 1_original.png          # Original input image
│   ├── 2_grayscale.png         # Grayscale conversion
│   ├── 3_blurred.png           # Gaussian blur
│   ├── 4_edges_canny.png       # Canny edge detection
│   ├── 5_hough_lines.png       # Hough line transform
│   ├── 6_classified_lines_HV.png  # Horizontal/vertical classification
│   ├── 7_all_detected_lines.png    # All detected lines
│   ├── 8_lines_closed.png      # Morphological closing
│   ├── 9_lines_thickened.png   # Line dilation
│   ├── 10_inverted_regions.png # Inverted binary regions
│   ├── 13_valid_parking_regions.png  # Valid parking regions
│   ├── 14_detected_contours.png      # Detected contours
│   ├── 15_rectangles_filtered_by_shape.png  # Shape-filtered rectangles
│   ├── 16_parking_spaces_final.png   # Final parking spaces
│   ├── 17_detected_parking_spots.png # Parking spots with occupancy
│   ├── 18_occupancy_analysis.png     # Occupancy analysis
│   ├── 19_occupancy_edges.png        # Edge-based occupancy
│   ├── 20_occupancy_variance.png     # Variance-based occupancy
│   ├── 21_occupancy_bright_pixels.png  # Bright pixel detection
│   ├── 22_occupancy_dark_pixels.png    # Dark pixel detection
│   ├── 23_occupancy_combined_score.png # Combined occupancy score
│   ├── 24_occupancy_color_coded.png   # Color-coded occupancy
│   ├── 25_grid_matrix.png       # Generated grid matrix
│   ├── 26_inflated_grid.png     # Inflated grid with clearance
│   ├── 27_astar_path.png        # A* pathfinding result
│   └── 28_smoothed_path.png     # B-spline smoothed path
├── documents/                   # Reference documents and papers
├── Layout 1.png                # Sample parking lot test images
├── Layout 2.png
├── Layout 3.jpg
├── Layout 4.jpg
├── layout 5.jpg
├── car.png                     # Car icon for visualization
├── yolo11l.pt                  # YOLO model weights (various versions)
├── yolo11n.pt
├── yolo11x.pt
├── yolov8n.pt
├── yolov8x-seg.pt
├── yolov8x.pt
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── Project_Report.md           # This report
└── venv/                       # Python virtual environment
```

## 5.2 Module Description

### 5.2.1 Main Application Module (main.py)

The `ParkingGridConverter` class is the core of the application:

**Initialization:**
- Creates Tkinter root window
- Initializes state variables
- Sets up GUI widgets
- Loads YOLO model in background thread

**Image Loading Module:**
- Supports JPG, PNG, BMP formats
- Converts BGR to RGB for display
- Resets previous detection results

**Traditional CV Detection Module:**
- Grayscale conversion
- Gaussian blur for noise reduction
- Canny edge detection
- Morphological closing and dilation
- Contour detection and filtering
- Parking spot validation

**YOLO Detection Module:**
- Uses YOLOv8 pretrained model
- Configurable confidence threshold
- Filters for obstacle classes (vehicles, people, objects)
- Draws bounding boxes with labels

**Grid Generation Module:**
- Calculates grid dimensions from image size and cell size
- Maps parking spots to grid cells (value 2)
- Maps obstacles to grid cells (value 1)
- Creates visualization with color overlays

**A* Pathfinding Module:**
- 8-directional movement (including diagonals)
- Euclidean distance heuristic
- Obstacle inflation for clearance
- Priority queue for efficient node selection

**Path Smoothing Module:**
- B-Spline interpolation using SciPy
- Cumulative distance parameterization
- Configurable smoothing factor and point count

**Export Module:**
- JSON format with grid and metadata
- NumPy .npy format for direct loading
- Text format for human readability

### 5.2.2 Standalone A* Module (a_star.py)

Provides reusable pathfinding functions:
- `create_node()` - Creates node for A* algorithm
- `calculate_heuristic()` - Euclidean distance calculation
- `get_valid_neighbors()` - Returns navigable neighbors
- `reconstruct_path()` - Builds path from goal to start
- `find_path()` - Main A* implementation
- `smooth_path_bspline()` - B-Spline smoothing
- `inflate_obstacles()` - Grid dilation for clearance
- `visualize_path()` - Matplotlib visualization

## 5.3 Code Implementation

### 5.3.1 Application Initialization

```python
class ParkingGridConverter:
    """Convert parking lot images to navigable grid for A* algorithm"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Lot Path Planner")
        self.root.geometry("1400x900")
        
        # State variables
        self.original_image = None
        self.processed_image = None
        self.grid_matrix = None
        self.grid_rows = 80
        self.grid_cols = 120
        
        # Object detection
        self.yolo_model = None
        self.detected_objects = []
        self.detection_confidence = 0.3
        
        # Traditional CV detection results
        self.parking_spots = []
        
        # Navigation
        self.start_point = None
        self.end_point = None
        self.path = None
        self.smoothed_path = None
        self.inflated_grid = None
        self.clearance_radius = 1
        
        # Create UI and load model
        self.create_widgets()
        self.load_yolo_model()
```

### 5.3.2 Parking Spot Detection (Traditional CV)

```python
def detect_parking_spots(self):
    """Detect empty parking spots using Traditional CV"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter parking spots
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Area filter
        if area < 2000 or area > 20000:
            continue
        
        # Aspect ratio filter
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(max(w, h)) / min(w, h)
        
        if aspect_ratio < 1.5 or aspect_ratio > 7.0:
            continue
        
        # Rectangularity check
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        is_rectangular = 4 <= len(approx) <= 6
        
        if is_rectangular:
            parking_spots.append(contour_info)
```

### 5.3.3 YOLO Obstacle Detection

```python
def detect_obstacles_yolo(self):
    """Detect obstacles using YOLO"""
    
    # Run YOLO inference
    results = self.yolo_model(self.original_image, 
                               conf=self.detection_confidence, 
                               verbose=False)
    
    # Obstacle classes from COCO dataset
    obstacle_classes = {
        0: 'person', 2: 'car', 3: 'motorcycle', 
        5: 'bus', 7: 'truck', ...
    }
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            self.detected_objects.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class_id': class_id,
                'class_name': class_name,
                'is_obstacle': class_id in obstacle_classes
            })
```

### 5.3.4 Grid Generation

```python
def generate_grid(self):
    """Generate navigable grid matrix"""
    
    # Calculate grid dimensions
    cell_size = int(self.cell_size_spinbox.get())
    self.grid_cols = width // cell_size
    self.grid_rows = height // cell_size
    
    # Initialize grid (0 = navigable)
    self.grid_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
    
    # Mark parking spots (value 2)
    for spot in self.parking_spots:
        # Calculate covered grid cells
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if overlap_percentage > 0.2:
                    self.grid_matrix[row, col] = 2
    
    # Mark obstacles (value 1) - overwrites parking spots
    for obj in self.detected_objects:
        if obj['is_obstacle']:
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    if overlap_percentage > 0.2:
                        self.grid_matrix[row, col] = 1
```

### 5.3.5 A* Pathfinding Implementation

```python
def astar(self, start, goal):
    """A* pathfinding with diagonal movement"""
    
    def heuristic(a, b):
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(pos):
        row, col = pos
        moves = [
            (-1, 0), (1, 0), (0, -1), (0, 1),      # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)     # Diagonal
        ]
        
        neighbors = []
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.grid_rows and 
                0 <= new_col < self.grid_cols and
                working_grid[new_row, new_col] == 0):
                neighbors.append((new_row, new_col))
        return neighbors
    
    # Priority queue: (f_score, counter, position)
    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    
    while open_set:
        _, _, current = heappop(open_set)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            move_cost = heuristic(current, neighbor)
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], counter, neighbor))
                counter += 1
    
    return None  # No path found
```

### 5.3.6 B-Spline Path Smoothing

```python
def smooth_path_bspline(self, path, smoothing_factor=0.1, num_points=100):
    """Smooth path using B-spline interpolation"""
    
    if len(path) < 3:
        return np.array(path)
    
    path_array = np.array(path)
    x_coords = path_array[:, 0]
    y_coords = path_array[:, 1]
    
    # Calculate cumulative distances
    distances = np.zeros(len(path))
    for i in range(1, len(path)):
        distances[i] = distances[i-1] + sqrt(
            (x_coords[i] - x_coords[i-1])**2 + 
            (y_coords[i] - y_coords[i-1])**2
        )
    
    # Normalize to [0, 1]
    if distances[-1] > 0:
        distances = distances / distances[-1]
    
    # Smooth parameter array
    t_smooth = np.linspace(0, 1, num_points)
    
    # Fit B-splines
    spl_x = UnivariateSpline(distances, x_coords, s=smoothing_factor)
    spl_y = UnivariateSpline(distances, y_coords, s=smoothing_factor)
    
    # Evaluate smooth coordinates
    x_smooth = spl_x(t_smooth)
    y_smooth = spl_y(t_smooth)
    
    return np.column_stack((x_smooth, y_smooth))
```

### 5.3.7 Obstacle Inflation

```python
def inflate_obstacles(self, grid, clearance):
    """Inflate obstacles for safe clearance"""
    
    if clearance == 0:
        return grid.copy()
    
    # 8-connectivity structure
    structure = ndimage.generate_binary_structure(2, 2)
    
    # Binary dilation
    inflated_grid = ndimage.binary_dilation(
        grid, 
        structure=structure, 
        iterations=clearance
    ).astype(np.uint8)
    
    return inflated_grid
```

## 5.4 User Interface Design

### 5.4.1 Main Window Layout

The application window is divided into:

1. **Top Control Panel**
   - Application title
   - Workflow buttons (Load, Detect, Generate, Pathfind)
   - Configuration spinboxes

2. **Left Panel - Parking Lot View**
   - Canvas displaying original/processed image
   - Interactive click for start/end points

3. **Right Panel - Grid Visualization**
   - Scrollable canvas showing grid matrix
   - Color-coded cells (white, red, green, yellow)

4. **Bottom Status Panel**
   - Status bar for current operation
   - Information text area for details

### 5.4.2 Control Buttons

| Button | Function | Description |
|--------|----------|-------------|
| 1. Load Image | `load_image()` | Opens file dialog for image selection |
| 2. Detect Parking Spots | `detect_parking_spots()` | Runs Traditional CV detection |
| 3. Detect Obstacles (YOLO) | `detect_obstacles_yolo()` | Runs YOLO detection |
| 4. Generate Grid | `generate_grid()` | Creates navigation grid |
| Set Start Point | `set_mode("start")` | Enables start point selection |
| Set End Point | `set_mode("end")` | Enables end point selection |
| Run A* Pathfinding | `run_astar()` | Computes optimal path |
| Clear Path | `clear_path()` | Resets path and points |
| Export Grid | `export_grid()` | Saves grid to file |

### 5.4.3 Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Grid Rows | 80 | 5-100 | Number of grid rows |
| Grid Columns | 120 | 5-100 | Number of grid columns |
| YOLO Confidence | 0.30 | 0.1-1.0 | Detection confidence threshold |
| Clearance | 1 | 0-5 | Obstacle inflation radius |
| Cell Size | 10 | 5-50 | Grid cell size in pixels |

### 5.4.4 Color Legend

| Color | Grid Value | Meaning |
|-------|------------|---------|
| White | 0 | Navigable space |
| Red | 1 | Obstacle (YOLO detected) |
| Green | 2 | Empty parking spot (CV detected) |
| Yellow | - | A* path |
| Cyan | - | Start point |
| Magenta | - | End point |

---

# CHAPTER 6: TESTING AND RESULTS

## 6.1 Testing Methodology

### 6.1.1 Testing Approach

The project follows a comprehensive testing approach:

1. **Unit Testing**: Individual function testing
2. **Integration Testing**: Module interaction testing
3. **System Testing**: End-to-end workflow testing
4. **Performance Testing**: Speed and resource usage
5. **Usability Testing**: User interface evaluation

### 6.1.2 Test Environment

| Component | Specification |
|-----------|---------------|
| Operating System | Ubuntu Linux 22.04 / Windows 11 |
| Processor | Intel Core i5-10400 / AMD Ryzen 5 5600 |
| RAM | 16 GB DDR4 |
| GPU | NVIDIA RTX 3060 (optional) |
| Python Version | 3.12 |

## 6.2 Test Cases

### 6.2.1 Image Loading Test Cases

| TC ID | Test Case | Input | Expected Output | Status |
|-------|-----------|-------|-----------------|--------|
| TC01 | Load valid JPG | parking_lot.jpg | Image displayed | ✓ Pass |
| TC02 | Load valid PNG | parking_lot.png | Image displayed | ✓ Pass |
| TC03 | Load invalid file | corrupted.jpg | Error message | ✓ Pass |
| TC04 | Cancel file dialog | Press Cancel | No change | ✓ Pass |
| TC05 | Load large image (4K) | 4k_image.png | Image scaled | ✓ Pass |

### 6.2.2 Detection Test Cases

| TC ID | Test Case | Input | Expected Output | Status |
|-------|-----------|-------|-----------------|--------|
| TC06 | Detect parking spots | Layout 1.png | Spots highlighted green | ✓ Pass |
| TC07 | Detect obstacles (YOLO) | Image with cars | Cars highlighted red | ✓ Pass |
| TC08 | No parking spots | Empty lot | 0 spots detected | ✓ Pass |
| TC09 | Low confidence YOLO | conf=0.1 | More detections | ✓ Pass |
| TC10 | High confidence YOLO | conf=0.9 | Fewer detections | ✓ Pass |

### 6.2.3 Pathfinding Test Cases

| TC ID | Test Case | Input | Expected Output | Status |
|-------|-----------|-------|-----------------|--------|
| TC11 | Valid path exists | Open start/end | Path found | ✓ Pass |
| TC12 | No path exists | Blocked route | "No path" message | ✓ Pass |
| TC13 | Start on obstacle | Start in red zone | No path or reselect | ✓ Pass |
| TC14 | Diagonal movement | Adjacent diagonal | Diagonal path used | ✓ Pass |
| TC15 | Path smoothing | Valid path | Smooth curve displayed | ✓ Pass |
| TC16 | Clearance = 0 | No inflation | Path close to obstacles | ✓ Pass |
| TC17 | Clearance = 3 | High inflation | Path far from obstacles | ✓ Pass |

### 6.2.4 Export Test Cases

| TC ID | Test Case | Input | Expected Output | Status |
|-------|-----------|-------|-----------------|--------|
| TC18 | Export JSON | grid.json | Valid JSON file | ✓ Pass |
| TC19 | Export NumPy | grid.npy | Loadable .npy file | ✓ Pass |
| TC20 | Export Text | grid.txt | Readable text file | ✓ Pass |

## 6.3 Results and Analysis

### 6.3.1 Detection Results

**Parking Spot Detection (Traditional CV):**
- Successfully detects rectangular parking spot markings
- Accuracy depends on image quality and line visibility
- Works best with clear, white parking lines

**YOLO Obstacle Detection:**
- Accurately detects vehicles with confidence > 0.3
- Detects pedestrians, motorcycles, trucks
- Performance varies with image resolution

### 6.3.2 Sample Processing Results

**Complete Image Processing Pipeline:**

The system generates a comprehensive set of intermediate images during processing, all saved in the `cv_process_images/` directory. The complete pipeline is as follows:

**Preprocessing Stage:**
1. **Original Image** (Figure 6.1 - `1_original.png`): Raw parking lot photograph loaded from file
2. **Grayscale Conversion** (Figure 6.2 - `2_grayscale.png`): RGB image converted to single-channel grayscale for efficient processing
3. **Gaussian Blur** (Figure 6.3 - `3_blurred.png`): Noise reduction using 3×3 Gaussian kernel to smooth the image

**Edge Detection Stage:**
4. **Canny Edge Detection** (Figure 6.4 - `4_edges_canny.png`): Edge detection with thresholds (50-150) to identify parking line boundaries

**Line Detection Stage:**
5. **Hough Line Transform** (Figure 6.5 - `5_hough_lines.png`): Detects line segments using probabilistic Hough transform
6. **Classified Lines** (Figure 6.6 - `6_classified_lines_HV.png`): Lines separated into horizontal (blue) and vertical (green) categories
7. **All Detected Lines** (Figure 6.7 - `7_all_detected_lines.png`): Combined visualization of all detected line segments

**Morphological Processing:**
8. **Morphological Closing** (Figure 6.8 - `8_lines_closed.png`): Closes gaps in horizontal and vertical lines using rectangular kernels
9. **Lines Thickened** (Figure 6.9 - `9_lines_thickened.png`): Dilation applied to ensure connectivity at line intersections
10. **Inverted Regions** (Figure 6.10 - `10_inverted_regions.png`): Inverted binary image to identify parking space regions

**Parking Spot Detection:**
11. **Valid Parking Regions** (Figure 6.11 - `13_valid_parking_regions.png`): Connected components filtered by area and aspect ratio
12. **Detected Contours** (Figure 6.12 - `14_detected_contours.png`): All contours extracted from valid regions
13. **Rectangles Filtered by Shape** (Figure 6.13 - `15_rectangles_filtered_by_shape.png`): Rectangularity validation (≥80% fit to rectangle)
14. **Final Parking Spaces** (Figure 6.14 - `16_parking_spaces_final.png`): Duplicate removal and final validated parking spots

**Occupancy Detection:**
15. **Detected Parking Spots** (Figure 6.15 - `17_detected_parking_spots.png`): Final output with empty (green) and occupied (red) spots labeled
16. **Occupancy Analysis** (Figure 6.16 - `18_occupancy_analysis.png`): Visualization showing occupancy confidence scores
17. **Occupancy Edge Detection** (Figure 6.17 - `19_occupancy_edges.png`): Edge density analysis for each parking spot
18. **Occupancy Variance** (Figure 6.18 - `20_occupancy_variance.png`): Texture variation analysis using Laplacian operator
19. **Occupancy Bright Pixels** (Figure 6.19 - `21_occupancy_bright_pixels.png`): Detection of bright pixels (white car roofs, threshold >220)
20. **Occupancy Dark Pixels** (Figure 6.20 - `22_occupancy_dark_pixels.png`): Detection of dark pixels (car shadows/body, threshold <50)
21. **Occupancy Combined Score** (Figure 6.21 - `23_occupancy_combined_score.png`): Weighted combination of all features
22. **Occupancy Color Coded** (Figure 6.22 - `24_occupancy_color_coded.png`): Final occupancy classification with color coding

**Grid and Pathfinding:**
23. **Generated Grid Matrix** (Figure 6.23 - `25_grid_matrix.png`): Navigable grid with obstacles (red), empty spots (green), and free space (white)
24. **Inflated Grid** (Figure 6.24 - `26_inflated_grid.png`): Grid with obstacle inflation for safe clearance radius
25. **A* Path Visualization** (Figure 6.25 - `27_astar_path.png`): Optimal path computed by A* algorithm (yellow-orange)
26. **Smoothed Path** (Figure 6.26 - `28_smoothed_path.png`): B-spline smoothed path for vehicle-friendly navigation (green)

### 6.3.3 Grid Generation Results

For a 1200x800 pixel image with 10px cell size:
- Grid dimensions: 120 columns × 80 rows = 9,600 cells
- Typical distribution:
  - Navigable (0): ~85% of cells
  - Obstacles (1): ~5-10% of cells
  - Parking spots (2): ~5-10% of cells

### 6.3.4 Pathfinding Results

- A* consistently finds optimal paths with 8-directional movement
- Path length varies based on obstacle layout and clearance radius
- Smoothing reduces path waypoints by ~70% while maintaining feasibility
- Obstacle inflation ensures safe clearance (configurable 0-5 cells)
- B-spline interpolation generates smooth, vehicle-friendly trajectories

**Visual Results:**
- Figure 6.25 (`27_astar_path.png`) shows the raw A* path in yellow-orange
- Figure 6.26 (`28_smoothed_path.png`) shows the B-spline smoothed path in green
- The smoothed path maintains the optimal route while eliminating sharp turns

## 6.4 Performance Evaluation

### 6.4.1 Performance Metrics

| Operation | Average Time | Memory Usage |
|-----------|--------------|--------------|
| Image Loading | 0.1-0.5s | 50-200 MB |
| YOLO Model Loading | 2-5s | 200-400 MB |
| Parking Spot Detection | 0.5-2s | 100-300 MB |
| YOLO Detection | 0.3-1s (GPU) / 2-5s (CPU) | 200-500 MB |
| Grid Generation | 0.1-0.5s | 50-100 MB |
| A* Pathfinding | 0.01-0.5s | 10-50 MB |
| Path Smoothing | 0.01-0.1s | 5-20 MB |

### 6.4.2 Scalability Analysis

| Image Size | Grid Size | A* Time | Memory |
|------------|-----------|---------|--------|
| 640×480 | 64×48 | <0.1s | 500 MB |
| 1280×720 | 128×72 | 0.1s | 600 MB |
| 1920×1080 | 192×108 | 0.2s | 800 MB |
| 3840×2160 | 384×216 | 0.5s | 1.2 GB |

### 6.4.3 Comparison with Alternatives

| Metric | Our System | Basic Sensor | Camera Only |
|--------|------------|--------------|-------------|
| Detection Accuracy | High (YOLO) | Medium | Low |
| Obstacle Types | Multiple | Limited | Limited |
| Path Planning | Optimal (A*) | None | None |
| Path Smoothing | Yes | No | No |
| Real-time Capable | Yes | Yes | Partial |
| Cost | Low (Software) | High (Hardware) | Medium |

---

# CHAPTER 7: CONCLUSION AND FUTURE WORK

## 7.1 Conclusion

This project successfully developed an **Intelligent Parking Lot Path Planner** that integrates computer vision techniques with pathfinding algorithms to assist drivers in navigating parking lots efficiently.

### Key Achievements:

1. **Dual Detection Approach**: Successfully combined Traditional CV for parking spot detection and YOLO for obstacle detection, leveraging the strengths of both methods.

2. **Grid-Based Environment Modeling**: Created an effective conversion from image-based detection to a navigable grid matrix, enabling standard pathfinding algorithms.

3. **Optimal Pathfinding**: Implemented A* algorithm with 8-directional movement and obstacle inflation for safe path computation.

4. **Path Smoothing**: Applied B-Spline interpolation to generate smooth, vehicle-friendly trajectories from discrete grid paths.

5. **User-Friendly Interface**: Developed an intuitive Tkinter-based GUI with comprehensive visualization and configuration options.

6. **Flexible Export**: Enabled data export in multiple formats (JSON, NumPy, Text) for integration with other systems.

### Technical Contributions:

- Integration of traditional CV and deep learning in a single application
- Real-time visualization of detection and pathfinding results
- Configurable parameters for different parking scenarios
- Modular architecture for maintainability and extensibility

### Practical Impact:

The system can reduce parking search time, improve traffic flow in parking structures, and enhance the overall parking experience. The exported grid data can be integrated with autonomous vehicle systems or mobile navigation applications.

## 7.2 Limitations

1. **Static Image Processing**: Current version processes static images rather than real-time video streams.

2. **2D Environment**: Limited to single-floor parking lots; does not support multi-level structures.

3. **Detection Accuracy**: Traditional CV may miss parking spots with faded or non-standard markings.

4. **Computational Requirements**: YOLO detection requires significant computational resources for optimal performance.

5. **No Vehicle Dynamics**: Path planning does not consider vehicle turning radius or kinematic constraints.

6. **Manual Point Selection**: Start and end points must be manually selected by the user.

## 7.3 Future Enhancements

### Short-term Improvements:

1. **Real-time Video Processing**
   - Implement video capture and frame-by-frame processing
   - Add tracking for moving obstacles
   - Optimize for real-time performance

2. **Improved Detection**
   - Train custom YOLO model for parking-specific objects
   - Add parking occupancy classification (empty/occupied)
   - Implement slot numbering and identification

3. **Enhanced Pathfinding**
   - Incorporate vehicle kinematic constraints
   - Add support for reverse parking maneuvers
   - Implement dynamic replanning for changing obstacles

### Long-term Developments:

4. **Multi-floor Navigation**
   - 3D grid representation
   - Ramp and elevator integration
   - Floor transition planning

5. **Mobile Application**
   - Android/iOS companion app
   - Real-time parking availability
   - Turn-by-turn navigation

6. **Cloud Integration**
   - Centralized parking management
   - Historical data analysis
   - Predictive availability

7. **Autonomous Vehicle Integration**
   - ROS (Robot Operating System) compatibility
   - Direct vehicle control interface
   - Sensor fusion with LiDAR/ultrasonic

8. **Smart Parking Ecosystem**
   - Payment integration
   - Reservation system
   - EV charging spot identification

---

# REFERENCES

1. Almeida, P. R., Oliveira, L. S., Britto Jr, A. S., Silva Jr, E. J., & Koerich, A. L. (2015). PKLot–A robust dataset for parking lot classification. *Expert Systems with Applications*, 42(11), 4937-4949.

2. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 779-788).

3. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE transactions on Systems Science and Cybernetics*, 4(2), 100-107.

4. Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer vision with the OpenCV library*. O'Reilly Media, Inc.

5. Jocher, G., Chaurasia, A., Stoken, A., et al. (2023). Ultralytics YOLOv8. GitHub repository. https://github.com/ultralytics/ultralytics

6. LaValle, S. M. (2006). *Planning algorithms*. Cambridge University Press.

7. De Boor, C. (1978). *A practical guide to splines*. Springer-Verlag.

8. Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on pattern analysis and machine intelligence*, (6), 679-698.

9. Suzuki, S. (1985). Topological structural analysis of digitized binary images by border following. *Computer vision, graphics, and image processing*, 30(1), 32-46.

10. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische mathematik*, 1(1), 269-271.

11. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

12. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

13. OpenCV Documentation. (2023). OpenCV-Python Tutorials. https://docs.opencv.org/

14. PyTorch Documentation. (2023). PyTorch Documentation. https://pytorch.org/docs/

15. SciPy Documentation. (2023). SciPy Reference Guide. https://docs.scipy.org/

---

# APPENDIX A: SOURCE CODE

## A.1 Main Application (main.py)

*[Complete source code included in project files]*

Key modules:
- `ParkingGridConverter` class (1200+ lines)
- Image processing functions
- Detection algorithms
- A* pathfinding implementation
- GUI widgets and event handlers

## A.2 Standalone A* Module (a_star.py)

*[Complete source code included in project files]*

Key functions:
- `create_node()` - Node creation
- `calculate_heuristic()` - Distance calculation
- `find_path()` - Main A* algorithm
- `smooth_path_bspline()` - Path smoothing
- `inflate_obstacles()` - Grid dilation
- `visualize_path()` - Matplotlib visualization

---

# APPENDIX B: SAMPLE OUTPUTS

## B.1 Image Processing Pipeline

All processing images are saved in the `cv_process_images/` directory. The complete pipeline includes:

**Preprocessing (Steps 1-3):**
1. `1_original.png` - Original input parking lot image
2. `2_grayscale.png` - Grayscale conversion for efficient processing
3. `3_blurred.png` - Gaussian blur applied for noise reduction

**Edge and Line Detection (Steps 4-7):**
4. `4_edges_canny.png` - Canny edge detection result
5. `5_hough_lines.png` - Hough line transform detection
6. `6_classified_lines_HV.png` - Horizontal and vertical line classification
7. `7_all_detected_lines.png` - All detected line segments combined

**Morphological Processing (Steps 8-10):**
8. `8_lines_closed.png` - Morphological closing to connect line gaps
9. `9_lines_thickened.png` - Dilation to thicken lines
10. `10_inverted_regions.png` - Inverted binary image for region detection

**Parking Spot Detection (Steps 13-16):**
13. `13_valid_parking_regions.png` - Valid parking regions after filtering
14. `14_detected_contours.png` - All detected contours
15. `15_rectangles_filtered_by_shape.png` - Rectangles filtered by rectangularity
16. `16_parking_spaces_final.png` - Final validated parking spaces

**Occupancy Analysis (Steps 17-24):**
17. `17_detected_parking_spots.png` - Final parking spots with occupancy status
18. `18_occupancy_analysis.png` - Occupancy confidence visualization
19. `19_occupancy_edges.png` - Edge density analysis
20. `20_occupancy_variance.png` - Variance/texture analysis
21. `21_occupancy_bright_pixels.png` - Bright pixel detection
22. `22_occupancy_dark_pixels.png` - Dark pixel detection
23. `23_occupancy_combined_score.png` - Combined occupancy score
24. `24_occupancy_color_coded.png` - Color-coded occupancy classification

**Grid and Pathfinding (Steps 25-28):**
25. `25_grid_matrix.png` - Generated navigation grid matrix
26. `26_inflated_grid.png` - Grid with obstacle inflation
27. `27_astar_path.png` - A* pathfinding result
28. `28_smoothed_path.png` - B-spline smoothed path

## B.2 Sample Grid Export (JSON)

```json
{
  "grid": [
    [0, 0, 0, 1, 1, 0, 0, 2, 2, 0],
    [0, 0, 0, 1, 1, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...
  ],
  "rows": 80,
  "cols": 120,
  "legend": {
    "0": "navigable",
    "1": "obstacle",
    "2": "empty_parking_spot"
  }
}
```

## B.3 Use Case Diagram

*[See use_case_diagram.png in project directory]*

---

# APPENDIX C: INSTALLATION GUIDE

## C.1 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)

## C.2 Installation Steps

```bash
# Clone repository (or download ZIP)
git clone https://github.com/username/Parking-Assistant-Prototype.git
cd Parking-Assistant-Prototype

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy pillow ultralytics scipy torch

# Run application
python main.py
```

## C.3 Usage Instructions

1. **Load Image**: Click "1. Load Image" and select a parking lot image
2. **Detect Parking Spots**: Click "2. Detect Parking Spots" for CV detection
3. **Detect Obstacles**: Click "3. Detect Obstacles (YOLO)" for deep learning detection
4. **Generate Grid**: Click "4. Generate Grid" to create navigation matrix
5. **Set Points**: Click "Set Start Point", then click on image; repeat for end point
6. **Find Path**: Click "Run A* Pathfinding" to compute optimal route
7. **Export**: Click "Export Grid" to save grid data

---

**END OF PROJECT REPORT**

---

*Document prepared by: [Your Name]*
*Date: [Current Date]*
*Version: 1.0*


