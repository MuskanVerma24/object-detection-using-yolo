import streamlit as st
import cv2
import numpy as np
import tempfile
import os


# CSS for app except footer
st.markdown(
    """
    <style>
    .stMainBlockContainer.ea3mdgi5{
        max-width:100%;
        padding: 0;
        margin: 0;
    }
    .video-container {
        position: relative;
        width: 100%;
        height: 90vh;
        overflow: hidden;
    }
    .video-container video {
        position: absolute;
        top: 0;
        left: 0;
        min-width: 100%;
        min-height: 100%;
        object-fit: cover;
        z-index: 1;
    }
    .video-container .animated-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 2;
        color: black;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        animation: fadeInOut 5s infinite;
        box-shadow:0 10px 20px 10px;;
        border-radius:20px;
        background:linear-gradient( transparent 10%,white);
        padding:0 15px 0 15px;
    }
    .decoration{
        text-align: center;
        text-decoration: overline;
        margin-top: 3%;
        font-size: 30px;
        font-weight: 800;
    }
    @keyframes fadeInOut {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(5){
        width: 75%;
        margin: auto;
    }
    .section-title {
        color:white;
        font-size: 30px;
        font-weight: bold;
        margin-top: 40px;
        width: 70%;
        margin: auto;
    }
    .section-subtitle {
        color:white;
        font-size: 18px;
        font-weight: normal;
        margin-top: -10px;
        width: 70%;
        margin: auto;
        margin-bottom:10px;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stHorizontalBlock.st-emotion-cache-ocqkz7.e1f1d6gn5{
        width: 70%;
        margin: auto;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stHorizontalBlock.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div > div > div.stImage.st-emotion-cache-kn8v7q.e115fcil2 > div > img{
        height: 270px;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(6) > div{
        width: 70%;
        margin: auto;
        height: auto;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(6) > div > div{
        margin: auto;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(7){
        width: 70%;
        margin: auto;
    } 
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(8) > div{
        width: 70%;
        margin: auto;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(9) > div{
        margin-left: 15%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# header video with animated text
st.markdown(
    """
    <div class="video-container">
        <video autoplay muted loop>
            <source src="https://videos.pexels.com/video-files/6981614/6981614-hd_1920_1080_30fps.mp4">
            Your browser does not support HTML5 video.
        </video>
        <div class="animated-text">Welcome to Object Detection App</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load YOLO model and class labels
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


# Function to process a single frame
def process_frame(frame, net, classes, output_layers):
    height, width = frame.shape[:2]

    # Resize frame
    resized_frame = cv2.resize(frame, (416, 416))
    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_text = f"{round(confidences[i] * 100, 2)}%"
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return frame


st.markdown("<div class='decoration'>Upload an image or video for object detection </div>",unsafe_allow_html=True)


# Load YOLO model
net, classes, output_layers = load_yolo()


# File Uploader
uploaded_file = st.file_uploader("Upload an Image or Video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save the uploaded file temporarily while preserving the file extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"temp_file{file_extension}")

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Determine if the file is an image or video
    if file_extension.lower() in [".jpg", ".jpeg", ".png"]:

        # Process Image
        image = cv2.imread(temp_file_path)
        processed_image = process_frame(image, net, classes, output_layers)

        # Display the processed image
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Detected Image")


    elif file_extension.lower() in [".mp4", ".avi", ".mov"]:

        # Process Video
        cap = cv2.VideoCapture(temp_file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video file path
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Progress bar
        progress = st.progress(0)
        current_frame = 0

        st.write("Processing video, please wait...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, net, classes, output_layers)
            out.write(processed_frame)
            current_frame += 1

            # Update progress bar
            progress.progress(int((current_frame / total_frames) * 100))

        cap.release()
        out.release()

        # Download link for the processed video
        st.success("Video processing complete!")
        with open(output_path, "rb") as f:
            st.download_button("Download Detected Video", f, file_name="detected_video.mp4")


# Example X-ray Images Section
st.markdown("<div class='section-title floating-element floating-element-4'>Examples</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
        st.image(r"https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/1446e76-f181-6047-4e73-8d8ba3c6a50e_object_detection_1.webp", use_container_width=True)
        
with col2:
        st.video("https://www.shutterstock.com/shutterstock/videos/1064002762/preview/stock-footage-autonomous-or-driverless-car-driving-through-a-crowded-street-in-los-angeles-computer-vision-with.webm")



# footer
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        width: 100%;
        background-color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
            Thank you for using this app! See you soon! 👋
    </div>
    """,
    unsafe_allow_html=True,
)