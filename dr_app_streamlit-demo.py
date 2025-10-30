import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from transformers import pipeline

# --------------------------
# App Title
# --------------------------
st.set_page_config(layout="wide")
st.title("RetinAlyze: Diabetic Retinopathy Detector")

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["Retina Image Detection", "Ask a Medical Question", "Comments"])

# --------------------------
# Load DR model
# --------------------------
@st.cache_resource
def load_dr_model():
    return load_model("DiabeticRetinopathy_FINAL.keras")

model = load_dr_model()

# --------------------------
# Load NLP pipeline (placeholder)
# --------------------------
@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        framework="pt"
    )

qa_pipeline = load_qa_pipeline()

# --------------------------
# FAQ fallback answers
# --------------------------
faq_answers = {
    "what is diabetic retinopathy": (
        "Diabetic retinopathy is a complication of diabetes that affects the retina, the light-sensitive tissue at the back of the eye. "
        "High blood sugar damages the tiny blood vessels, leading to leakage, swelling, or abnormal vessel growth. "
        "Early stages may have no symptoms, but as the disease progresses, it can cause blurred vision, floaters, and even blindness if untreated."
    ),
    "what does class 1 diagnosis mean": (
        "Class 1 (Mild Non-Proliferative Diabetic Retinopathy) indicates early signs of retinal damage. "
        "Tiny microaneurysms appear in the retina, but vision is usually not affected at this stage. "
        "Regular monitoring is important to prevent progression."
    ),
    "what does class 2 diagnosis mean": (
        "Class 2 (Moderate Non-Proliferative) shows more blood vessel blockages and some retinal swelling. "
        "Vision may start to be affected, and patients should follow up closely with their ophthalmologist."
    ),
    "what does class 3 diagnosis mean": (
        "Class 3 (Severe Non-Proliferative) represents significant retinal vessel damage. "
        "There is a higher risk of vision loss, and more intensive monitoring or treatment may be needed."
    ),
    "what does class 4 diagnosis mean": (
        "Class 4 (Pre-Proliferative) is advanced; large areas of the retina have blocked vessels, causing oxygen deprivation. "
        "This can quickly progress to proliferative stages if untreated."
    ),
    "what does class 5 diagnosis mean": (
        "Class 5 (Proliferative Diabetic Retinopathy) is the most severe stage. "
        "New, abnormal blood vessels grow on the retina or optic nerve, greatly increasing the risk of severe vision loss. "
        "Timely treatment is critical to prevent blindness."
    ),
}

# --------------------------
# Retina Image Detection Tab
# --------------------------
with tabs[0]:
    st.header("Upload Retinal Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Retinal Image", use_container_width=True)

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        resize = tf.image.resize(img_array, (256, 256))
        yhat = model.predict(np.expand_dims(resize/255, 0))

        # Prediction class
        class_labels = ["Class 5", "Class 4", "Class 3", "Class 2", "Class 1"]
        pred_class_index = np.argmax(yhat)
        pred_confidence = yhat[0, pred_class_index]
        predicted_class = class_labels[pred_class_index]

        st.markdown("### Model Prediction")
        if pred_confidence >= 0.5:
            st.success(f"**POSITIVE** | Confidence: {pred_confidence:.2f}")
        else:
            st.warning(f"**NEGATIVE** | Confidence: {pred_confidence:.2f}")

        if pred_confidence >= 0.8:
            st.info(f"Predicted class: {predicted_class} (high confidence)")
        elif pred_confidence >= 0.6:
            st.info(f"Predicted class: {predicted_class} (moderately high confidence)")
        elif pred_confidence >= 0.4:
            st.info(f"Predicted class: {predicted_class} (moderate confidence)")
        elif pred_confidence >= 0.2:
            st.info(f"Predicted class: {predicted_class} (low confidence)")
        else:
            st.info(f"Predicted class: {predicted_class} (very low confidence)")

# --------------------------
# Medical Question Tab
# --------------------------
with tabs[1]:
    st.header("Ask a Medical Question")
    user_question = st.text_input("Type your question here:")

    if user_question:
        key = user_question.strip().lower()
        answer = faq_answers.get(key)
        if answer:
            st.markdown(f"**Answer:** {answer}")
        else:
            try:
                result = qa_pipeline(
                    question=user_question,
                    context="Diabetic retinopathy is a complication of diabetes affecting the eyes. It damages retinal blood vessels and can lead to vision loss if untreated."
                )
                st.markdown(f"**Answer:** {result['answer']}")
            except Exception:
                st.markdown("The model did not generate an answer. Try asking a simpler question.")

# --------------------------
# Comments Tab
# --------------------------
with tabs[2]:
    st.header("Comments")
    if "comments" not in st.session_state:
        st.session_state.comments = []

    # Name and Comment fields
    name = st.text_input("Name:")
    comment_text = st.text_input("Comment:")
    if st.button("Post Comment"):
        if name and comment_text:
            st.session_state.comments.append(f"{name}: {comment_text}")
            st.success("Comment added!")

    if st.session_state.comments:
        st.subheader("All Comments:")
        for c in st.session_state.comments:
            st.markdown(f"- {c}")