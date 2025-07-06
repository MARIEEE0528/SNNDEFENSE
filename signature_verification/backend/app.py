from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import cv2
import pickle
import numpy as np
import tensorflow as tf
import uuid

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from preprocess import preprocess_signature
from model import load_siamese_model, load_enhanced_siamese_model
from utils import verify_signature

app = Flask(__name__, template_folder="../frontend/templates")
STATIC_TEMP = "static/temp"
os.makedirs(STATIC_TEMP, exist_ok=True)

# Load reference embeddings
with open("enhanced_reference_embeddings.pkl", "rb") as f:
    enhanced_reference_embeddings = pickle.load(f)

with open("base_reference_embeddings.pkl", "rb") as f:
    base_reference_embeddings = pickle.load(f)

# Load models
base_siamese_model = load_siamese_model("../../models/base_CEDAR_siamese_model.keras")
base_model = base_siamese_model.get_layer("base_network")

enhanced_siamese_model = load_enhanced_siamese_model("../../enhanced_weights/enhanced_CEDAR.keras")
enhanced_model = enhanced_siamese_model.get_layer("base_network")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_writers", methods=["GET"])
def get_writers():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/cedar"))
    writer_ids = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])
    return jsonify(writer_ids)

@app.route("/base", methods=["POST"])
def verify_base():
    uploaded_file = request.files["signature"]
    claimed_writer_id = request.form["writer_id"]

    # Save uploaded file
    raw_path = f"{STATIC_TEMP}/{uploaded_file.filename}"
    minmax_path = f"{STATIC_TEMP}/minmax_{uploaded_file.filename}"
    uploaded_file.save(raw_path)

    # Preprocess with MinMax and get edge count
    minmax_img, minmax_edge_count = preprocess_signature(raw_path, preprocessing_type="minmax")
    cv2.imwrite(minmax_path, minmax_img.squeeze() * 255)

    # Predict and normalize embedding
    raw_emb = base_model.predict(np.expand_dims(minmax_img, axis=0), verbose=0)[0].flatten()
    uploaded_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-10)

    # Find closest writer across all references
    min_global_dist = float("inf")
    closest_writer = None
    for writer, refs in base_reference_embeddings.items():
        for ref in refs:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_global_dist:
                min_global_dist = dist
                closest_writer = writer

    # Get claimed writer's distance (anchor–positive)
    min_pos_dist = float("inf")
    if claimed_writer_id in base_reference_embeddings:
        for ref in base_reference_embeddings[claimed_writer_id]:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_pos_dist:
                min_pos_dist = dist
    else:
        min_pos_dist = float("inf")  # If claimed writer ID doesn't exist

    # Apply threshold
    threshold = 0.4982339
    distance = min_pos_dist
    accepted = distance <= threshold
    result = "Genuine" if accepted else "Forged"

    # True if both distance is low AND writer is correct
    is_authentic = (closest_writer == claimed_writer_id) and (distance <= threshold)

    # Now define rejection_type clearly
    if is_authentic:
        rejection_type = "true_accept"
    elif (closest_writer != claimed_writer_id) and (distance <= threshold):
        rejection_type = "false_acceptance"
    elif (closest_writer == claimed_writer_id) and (distance > threshold):
        rejection_type = "false_rejection"
    else:
        rejection_type = "true_reject"

    # Log to terminal and CSV
    print(f"[BASE] Writer: {claimed_writer_id}, File: {uploaded_file.filename}, MinMax Edge Count: {minmax_edge_count}")
    with open("edge_counts.csv", "a") as f:
        f.write(f"base,{claimed_writer_id},{uploaded_file.filename},minmax,{minmax_edge_count}\n")
    with open("rejection_logs.csv", "a") as f:
        f.write(f"base,{claimed_writer_id},{uploaded_file.filename},{distance:.4f},{threshold:.4f},{closest_writer},{rejection_type}\n")

    # Build response
    result = {
        "result": "genuine" if distance <= threshold else "forged",
        "identity_match": closest_writer == claimed_writer_id,
        "distance": float(round(distance, 4)),
        "threshold": float(threshold),
        "confidence": float(round(1 - (distance / threshold), 4)) if distance <= threshold else float(round(distance / threshold, 4)),
        "raw_image_url": f"/static/temp/{uploaded_file.filename}",
        "minmax_image_url": f"/static/temp/minmax_{uploaded_file.filename}",
        "clahe_image_url": "",  # (CLAHE only for enhanced)
        "minmax_edge_count": minmax_edge_count,
        "closest_writer": closest_writer,
        "rejection_type": rejection_type,
        "claimed_writer_id": claimed_writer_id
    }

    return jsonify(result)

@app.route("/enhanced", methods=["POST"])
def verify_enhanced():
    uploaded_file = request.files["signature"]
    claimed_writer_id = request.form["writer_id"]

    raw_path = f"{STATIC_TEMP}/{uploaded_file.filename}"
    clahe_path = f"{STATIC_TEMP}/clahe_{uploaded_file.filename}"
    minmax_path = f"{STATIC_TEMP}/minmax_{uploaded_file.filename}"
    uploaded_file.save(raw_path)

    # CLAHE preprocessing + save
    clahe_img, clahe_edge_count = preprocess_signature(raw_path, preprocessing_type="clahe")
    cv2.imwrite(clahe_path, clahe_img.squeeze() * 255)

    # MinMax preprocessing (only for edge count display)
    minmax_img, minmax_edge_count = preprocess_signature(raw_path, preprocessing_type="minmax")
    cv2.imwrite(minmax_path, (minmax_img.squeeze() * 255).astype(np.uint8))

    # Get normalized embedding
    raw_emb = enhanced_model.predict(np.expand_dims(clahe_img, axis=0), verbose=0)[0].flatten()
    uploaded_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-10)

    # Find anchor–positive (claimed writer)
    min_pos_dist, pos_path = float("inf"), None
    for ref in enhanced_reference_embeddings[claimed_writer_id]:
        ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
        dist = np.linalg.norm(uploaded_emb - ref_emb)
        if dist < min_pos_dist:
            min_pos_dist = dist
            pos_path = ref["path"]

    # Find anchor–negative (other writers)
    min_neg_dist, neg_path = float("inf"), None
    for other_writer, refs in enhanced_reference_embeddings.items():
        if other_writer == claimed_writer_id:
            continue
        for ref in refs:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_neg_dist:
                min_neg_dist = dist
                neg_path = ref["path"]

    # Determine closest writer overall
    min_global_dist, closest_writer = float("inf"), None
    for writer, refs in enhanced_reference_embeddings.items():
        for ref in refs:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_global_dist:
                min_global_dist = dist
                closest_writer = writer

    # Thresholding logic (uses anchor–positive)
    threshold = 0.827
    distance = min_pos_dist

    # True if both distance is low AND writer is correct
    is_authentic = (closest_writer == claimed_writer_id) and (distance <= threshold)

    # Now define rejection_type clearly
    if is_authentic:
        rejection_type = "true_accept"
    elif (closest_writer != claimed_writer_id) and (distance <= threshold):
        rejection_type = "false_acceptance"
    elif (closest_writer == claimed_writer_id) and (distance > threshold):
        rejection_type = "false_rejection"
    else:
        rejection_type = "true_reject"

    # Logs
    with open("edge_counts.csv", "a") as f:
        f.write(f"enhanced,{claimed_writer_id},{uploaded_file.filename},clahe,{clahe_edge_count}\n")
        f.write(f"enhanced,{claimed_writer_id},{uploaded_file.filename},minmax,{minmax_edge_count}\n")

    with open("verification_logs.csv", "a") as f:
        f.write(f"enhanced,{claimed_writer_id},{uploaded_file.filename},{distance:.4f},{threshold:.4f},{closest_writer},{rejection_type}\n")

    # Return JSON response
    return jsonify({
        "result": "genuine" if distance <= threshold else "forged",
        "identity_match": closest_writer == claimed_writer_id,
        "distance": float(round(distance, 4)),
        "threshold": float(threshold),
        "positive_distance": float(round(min_pos_dist, 4)),
        "negative_distance": float(round(min_neg_dist, 4)),
        "positive_image_url": f"/{pos_path}",
        "negative_image_url": f"/{neg_path}",
        "closest_writer": closest_writer,
        "rejection_type": rejection_type,
        "raw_image_url": f"/static/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/static/temp/clahe_{uploaded_file.filename}",
        "minmax_image_url": f"/static/temp/minmax_{uploaded_file.filename}",
        "clahe_edge_count": clahe_edge_count,
        "minmax_edge_count": minmax_edge_count,
        "claimed_writer_id": claimed_writer_id
    })

    return jsonify(result)

@app.route("/get_triplet_example", methods=["POST"])
def get_triplet_example():
    try:
        uploaded_file = request.files["signature"]
        writer_id = request.form["writer_id"]
        filename = request.form.get("filename", f"anchor_{uuid.uuid4().hex}.png")
        anchor_path = os.path.join(STATIC_TEMP, filename)
        uploaded_file.save(anchor_path)

        # ✅ Fix: Unpack the tuple
        anchor_img, _ = preprocess_signature(anchor_path, preprocessing_type="clahe")

        # ✅ Use only the image for model prediction
        raw_emb = enhanced_model.predict(np.expand_dims(anchor_img, axis=0), verbose=0)[0].flatten()
        anchor_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-10)  # L2 normalize

        # Determine closest writer overall
        min_global_dist, closest_writer = float("inf"), None
        for writer, refs in enhanced_reference_embeddings.items():
            for ref in refs:
                ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
                dist = np.linalg.norm(anchor_emb - ref_emb)
                if dist < min_global_dist:
                    min_global_dist = dist
                    closest_writer = writer

        # Find closest positive (same writer)
        min_pos_dist, pos_path = float("inf"), None
        for ref in enhanced_reference_embeddings[writer_id]:
            ref_emb = ref["embedding"]
            ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
            dist = np.linalg.norm(anchor_emb - ref_emb)
            if dist < min_pos_dist:
                min_pos_dist = dist
                pos_path = ref["path"]

        # Find closest negative (different writer)
        min_neg_dist, neg_path = float("inf"), None
        for other_writer, refs in enhanced_reference_embeddings.items():
            if other_writer == writer_id:
                continue
            for ref in refs:
                ref_emb = ref["embedding"]
                ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
                dist = np.linalg.norm(anchor_emb - ref_emb)
                if dist < min_neg_dist:
                    min_neg_dist = dist
                    neg_path = ref["path"]

        # Fix macOS path to Windows path
        def fix_path(path):
           return path.replace(
        "/Users/christelle/Desktop/SNNDEFENSE",
        "C:/Users/Marie/SNN/SNNDEFENSE-1"
        )            

        def save_preview_image(src, dest_name):
            src = fix_path(src)
            if not os.path.exists(src):
               print(f"[ERROR] Image path does not exist: {src}")
               return None  # or raise an exception or return a fallback image path
            
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

            if img is None:
               print(f"[ERROR] Failed to load image: {src}")
               return None

            if img.dtype != 'uint8':
               print("[WARNING] Converting image to uint8")
               img = img.astype('uint8')

            dest = os.path.join(STATIC_TEMP, dest_name)
            cv2.imwrite(dest, img)
            return f"/static/temp/{dest_name}"

        # ✅ Convert CLAHE-enhanced anchor back to uint8 for preview
        clahe_anchor_img = (anchor_img * 255).astype("uint8")
        anchor_dest = os.path.join(STATIC_TEMP, "triplet_anchor.png")
        cv2.imwrite(anchor_dest, clahe_anchor_img)

        anchor_url = "/static/temp/triplet_anchor.png"
        positive_url = save_preview_image(pos_path, "triplet_positive.png")
        negative_url = save_preview_image(neg_path, "triplet_negative.png")

        return jsonify({
            "anchor_url": anchor_url,
            "positive_url": positive_url,
            "negative_url": negative_url,
            "anchor_positive_dist": float(round(min_pos_dist, 4)),
            "anchor_negative_dist": float(round(min_neg_dist, 4)),
            "claimed_writer_id": writer_id,
            "closest_writer": closest_writer,
        })

    except Exception as e:
        print("[ERROR] Triplet generation failed:", e)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/verify_pair", methods=["POST"])
def verify_pair_signature():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    if writer_id not in base_reference_embeddings:
        return jsonify({"error": "Writer ID not found."}), 400

    raw_path = os.path.join(STATIC_TEMP, uploaded_file.filename)
    uploaded_file.save(raw_path)

    # ✅ MinMax preprocessing
    processed_img, _ = preprocess_signature(raw_path, preprocessing_type="minmax")
    cv2.imwrite(
        os.path.join(STATIC_TEMP, f"pair_uploaded_{uploaded_file.filename}"),
        (processed_img.squeeze() * 255).astype(np.uint8)
    )

    # ✅ Predict embedding and normalize
    raw_emb = base_model.predict(np.expand_dims(processed_img, axis=0), verbose=0)[0].flatten()
    uploaded_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-10)

    # ✅ Global closest (embedding-based) — SOP 2 insight
    min_global_dist = float("inf")
    closest_writer = None
    for writer, refs in base_reference_embeddings.items():
        for ref in refs:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_global_dist:
                min_global_dist = dist
                closest_writer = writer

    # ✅ Claimed writer comparison (anchor–positive)
    min_dist = float("inf")
    closest_ref = None
    for ref in base_reference_embeddings[writer_id]:
        ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
        dist = np.linalg.norm(uploaded_emb - ref_emb)
        if dist < min_dist:
            min_dist = dist
            closest_ref = ref["path"]

    # Fix macOS path to Windows path
    closest_ref = closest_ref.replace(
    "/Users/christelle/Desktop/SNNDEFENSE",
    "C:/Users/Marie/SNN/SNNDEFENSE-1"
    )
    
    # ✅ Save closest reference image
    ref_img = cv2.imread(closest_ref, cv2.IMREAD_GRAYSCALE)
   
    if ref_img is None:
      print(f"[ERROR] Failed to load image: {closest_ref}")
      return jsonify({"error": "Reference image not found or unreadable."}), 400
    
    if ref_img.dtype != 'uint8':
      print("[WARNING] Converting image to uint8")
    ref_img = ref_img.astype('uint8')

    ref_preview_name = f"pair_reference_{os.path.basename(closest_ref)}"
    cv2.imwrite(os.path.join(STATIC_TEMP, ref_preview_name), ref_img)

    return jsonify({
        "distance": round(float(min_dist), 4),
        "uploaded_image_url": f"/static/temp/pair_uploaded_{uploaded_file.filename}",
        "reference_image_url": f"/static/temp/{ref_preview_name}",
        "closest_writer": closest_writer,
        "claimed_writer_id": writer_id,
    })


@app.route("/static/temp/<path:filename>")
def serve_temp(filename):
    return send_from_directory(STATIC_TEMP, filename)

if __name__ == "__main__":
    app.run(debug=True)
    