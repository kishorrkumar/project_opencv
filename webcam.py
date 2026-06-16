import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

st.set_page_config(page_title="MediaPipe Face Glow", layout="centered")

st.title("🧬 AI MediaPipe Face Contour App")
st.write("Take a picture to instantly map a futuristic glowing matrix grid onto your facial structures!")

# Initialize MediaPipe Face Mesh solutions
mp_face_mesh = mp.solutions.face_mesh

# Define indices for specific facial paths to draw glowing lines
# (MediaPipe canonical face mapping indexes)
EYEBROW_LEFT = [70, 63, 105, 66, 107]
EYEBROW_RIGHT = [336, 296, 334, 293, 300]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def draw_glow_lines(img, landmarks, landmark_indices, color, thickness=3):
    """Draws smoothly connected anti-aliased glowing lines along landmark paths."""
    h, w, _ = img.shape
    points = []
    
    # Extract coordinate positions scaled back to image pixel size
    for idx in landmark_indices:
        landmark = landmarks[idx]
        pt = (int(landmark.x * w), int(landmark.y * h))
        points.append(pt)
        
    # Draw lines sequentially
    for i in range(len(points) - 1):
        # A layer of wider, blurrier color mimics an emissive glow effect
        cv2.line(img, points[i], points[i+1], color, thickness + 4, cv2.LINE_AA)
        # White/Bright core line
        cv2.line(img, points[i], points[i+1], (255, 255, 255), thickness, cv2.LINE_AA)

def apply_mediapipe_filter(img, glow_color):
    """Processes image via MediaPipe Face Mesh and paints neon filters."""
    # Convert RGB to BGR for MediaPipe processing pipeline
    rgb_for_mp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Run the Face Mesh model in static image mode
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(rgb_for_mp)
        output_img = img.copy()
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Draw specific futuristic geometric shapes mapped to your face
            draw_glow_lines(output_img, face_landmarks, FACE_OVAL, glow_color, thickness=2)
            draw_glow_lines(output_img, face_landmarks, EYEBROW_LEFT, glow_color, thickness=3)
            draw_glow_lines(output_img, face_landmarks, EYEBROW_RIGHT, glow_color, thickness=3)
            draw_glow_lines(output_img, face_landmarks, LIPS_OUTER, glow_color, thickness=2)
            
            return output_img, True
        else:
            return img, False

# --- UI Layout ---

# Let user pick their Sci-Fi laser color scheme
glow_choice = st.selectbox(
    "Choose your Neon Cyberpunk Theme:",
    ["Cyber Cyan", "Laser Pink", "Matrix Green", "Electric Yellow"]
)

# Color maps (RGB Tuples)
color_map = {
    "Cyber Cyan": (0, 242, 254),
    "Laser Pink": (255, 0, 127),
    "Matrix Green": (0, 255, 70),
    "Electric Yellow": (255, 234, 0)
}
selected_rgb = color_map[glow_choice]

picture = st.camera_input("Look straight ahead and click 'Take Photo'")

if picture is not None:
    # Convert the picture to OpenCV-readable numpy format
    file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    # Maintain proper RGB array structure
    rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    with st.spinner("Analyzing structural landmarks..."):
        processed_img, success = apply_mediapipe_filter(rgb_img, selected_rgb)
        
    if success:
        st.success("Successfully mapped 468 3D landmarks!")
    else:
        st.warning("No face detected. Ensure your face is fully lit and visible in frame.")

    # Display Output Canvas
    st.subheader("🤖 AI Enhanced Hologram")
    st.image(processed_img, use_container_width=True)
    
    # Streamlit Download Handling
    result_pil = Image.fromarray(processed_img)
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    st.download_button(
        label="📥 Download Cyberpunk Portrait",
        data=img_byte_arr,
        file_name="mediapipe_glow.jpg",
        mime="image/jpeg"
    )