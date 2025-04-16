from models import mtcnn_wrapper, viola_jones
import os
import matplotlib.pyplot as plt
import numpy as np
import time

def compare():
    detected_faces_mtcnn = {}
    detected_faces_viola_jones = {}
    directory = os.fsencode("images")
    for d in os.listdir(directory):
        sub_directory = os.path.join(directory, d)
        if not os.path.isdir(sub_directory):
            continue
        
        detected_faces_mtcnn[d.decode('utf-8')] = [0, 0]
        detected_faces_viola_jones[d.decode('utf-8')] = [0, 0]

        mtcnn_time = 0
        viola_jones_time = 0

        for img in os.listdir(sub_directory):
            img_path = os.path.join(sub_directory, img)

            start = time.time()
            num_faces_mtcnn = mtcnn_wrapper.mtcnn_face_detect(img_path, False)
            end = time.time()
            mtcnn_time += end - start

            start = time.time()
            num_faces_viola_jones = viola_jones.viola_jones_face_detect(img_path, False)
            end = time.time()
            viola_jones_time += end - start

            if num_faces_mtcnn >= 1:
                detected_faces_mtcnn[d.decode('utf-8')][0] += 1
                detected_faces_mtcnn[d.decode('utf-8')][1] += num_faces_mtcnn - 1

            if num_faces_viola_jones >= 1:
                detected_faces_viola_jones[d.decode('utf-8')][0] += 1
                detected_faces_viola_jones[d.decode('utf-8')][1] += num_faces_viola_jones - 1
 
    print(f"MTCNN needed time: {mtcnn_time}")
    print(f"Viola-Jones needed time: {viola_jones_time}")
    show_graphs(detected_faces_mtcnn, True)
    show_graphs(detected_faces_viola_jones, False)

def extract_data(detected_faces: dict):
    features = list(detected_faces.keys())
    num_faces = [v[0] for v in detected_faces.values()]
    false_positives = [v[1] for v in detected_faces.values()]
    return features, num_faces, false_positives

def by_angle(detected_faces: dict):
    by_angle = {}
    by_angle["degree_0"] = detected_faces["degree_0"]
    by_angle["degree_30"] = detected_faces["degree_30"]
    by_angle["degree_45"] = detected_faces["degree_45"]
    by_angle["degree_90"] = detected_faces["degree_90"]
    by_angle["looking_up"] = detected_faces["looking_up"]
    by_angle["looking_down"] = detected_faces["looking_down"]
    return by_angle

def by_lighting(detected_faces: dict):
    by_lighting = {}
    by_lighting["light"] = detected_faces["light"]
    by_lighting["dark"] = detected_faces["dark"]
    by_lighting["blurry"] = detected_faces["blurry"]
    return by_lighting

def by_missing_feature(detected_faces: dict): 
    by_missing_feature = {}
    by_missing_feature["missing_forehead"] = detected_faces["missing_forehead"]
    by_missing_feature["missing_eyebrows"] = detected_faces["missing_eyebrows"]
    by_missing_feature["missing_one_eye"] = detected_faces["missing_one_eye"]
    by_missing_feature["missing_eyes"] = detected_faces["missing_eyes"]
    by_missing_feature["missing_nose"] = detected_faces["missing_nose"]
    by_missing_feature["missing_eyes_and_nose"] = detected_faces["missing_eyes_and_nose"]
    by_missing_feature["missing_mouth"] = detected_faces["missing_mouth"]
    return by_missing_feature

def by_mimic(detected_faces: dict):
    by_mimic = {}
    by_mimic["smiling"] = detected_faces["smiling"]
    by_mimic["frowning"] = detected_faces["frowning"]
    return by_mimic

def show_graphs(detected_faces: dict, is_mtcnn: bool):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
    width = 0.35
    fig.canvas.manager.set_window_title("MTCNN Face Detection" if is_mtcnn else "Viola-Jones Face Detection")
    fig.suptitle("MTCNN Face Detection" if is_mtcnn else "Viola-Jones Face Detection")

    features, num_faces, false_positives = extract_data(by_angle(detected_faces))
    x = np.arange(len(features))

    ax1.bar(x - width/2, num_faces, width, label='Detected Faces', color='skyblue')
    ax1.bar(x + width/2, false_positives, width, label='False Positives', color='salmon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features)
    ax1.set_title("Face Detection by Angle")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    features, num_faces, false_positives = extract_data(by_lighting(detected_faces))
    x = np.arange(len(features))

    ax2.bar(x - width/2, num_faces, width, label='Detected Faces', color='skyblue')
    ax2.bar(x + width/2, false_positives, width, label='False Positives', color='salmon')
    ax2.set_xticks(x)
    ax2.set_xticklabels(features)
    ax2.set_title("Face Detection by Lighting")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    features, num_faces, false_positives = extract_data(by_missing_feature(detected_faces))
    x = np.arange(len(features))

    ax3.bar(x - width/2, num_faces, width, label='Detected Faces', color='skyblue')
    ax3.bar(x + width/2, false_positives, width, label='False Positives', color='salmon')
    ax3.set_xticks(x)
    ax3.set_xticklabels(features)
    ax3.set_title("Face Detection by Missing Features")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    features, num_faces, false_positives = extract_data(by_mimic(detected_faces))
    x = np.arange(len(features))

    ax4.bar(x - width/2, num_faces, width, label='Detected Faces', color='skyblue')
    ax4.bar(x + width/2, false_positives, width, label='False Positives', color='salmon')
    ax4.set_xticks(x)
    ax4.set_xticklabels(features)
    ax4.set_title("Face Detection by Mimic")
    ax4.set_ylabel("Count")
    ax4.legend()
    ax4.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
