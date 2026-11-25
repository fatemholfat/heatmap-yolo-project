
import cv2
import numpy as np
import json
import os
import time
import threading
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "botsort_reid.yaml"

CONFIDENCE = 0.35
IMAGE_SIZE = 960
MIN_HITS = 3
EMBED_INTERVAL = 15
SIMILARITY_THRESHOLD = 0.82

SAVE_VIDEO = True
OUTPUT_DIR = "output"
DATABASE_DIR = "database"
SHOW_WINDOWS = True

SOURCES = [
    {
        "name": "camera_1",
        "source": "input.mp4",
        "is_live": False
    }
]


class GlobalIdentityManager:

    def __init__(self):
        self.lock = threading.Lock()
        self.current_day = datetime.now().strftime("%Y-%m-%d")
        self.people = {}
        self.next_id = 1
        os.makedirs(DATABASE_DIR, exist_ok=True)
        self.load_today()

    def database_path(self):
        return os.path.join(
            DATABASE_DIR,
            f"people_{self.current_day}.json"
        )

    def reset_if_new_day(self):
        today = datetime.now().strftime("%Y-%m-%d")

        if today != self.current_day:
            self.current_day = today
            self.people = {}
            self.next_id = 1
            print("\nNew day detected. Daily count reset.")

    def load_today(self):
        path = self.database_path()

        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.people = data.get("people", {})
            self.next_id = data.get("next_id", 1)

            print("Today's identity memory loaded.")

        except Exception as error:
            print("Database loading error:", error)

    def save(self):
        path = self.database_path()
        temp_path = path + ".tmp"

        data = {
            "people": self.people,
            "next_id": self.next_id
        }

        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

        os.replace(temp_path, path)

    @staticmethod
    def normalize(vector):
        vector = np.asarray(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vector)

        if norm == 0:
            return None

        return vector / norm

    def identify(self, embedding, forbidden_ids=None):
        embedding = self.normalize(embedding)

        if embedding is None:
            return None, False

        forbidden_ids = forbidden_ids or set()

        with self.lock:
            self.reset_if_new_day()

            best_id = None
            best_similarity = -1.0

            for person_id, data in self.people.items():

                if int(person_id) in forbidden_ids:
                    continue

                old_embedding = self.normalize(data["embedding"])

                if old_embedding is None:
                    continue

                if old_embedding.shape != embedding.shape:
                    continue

                similarity = float(np.dot(embedding, old_embedding))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = person_id

            if (
                best_id is not None
                and best_similarity >= SIMILARITY_THRESHOLD
            ):
                old = self.people[best_id]

                old_embedding = self.normalize(old["embedding"])

                updated = (
                    0.9 * old_embedding +
                    0.1 * embedding
                )

                updated = self.normalize(updated)

                old["embedding"] = updated.tolist()
                old["last_seen"] = time.time()
                old["seen_count"] = old.get("seen_count", 0) + 1

                self.save()

                return int(best_id), False

            new_id = self.next_id
            self.next_id += 1

            self.people[str(new_id)] = {
                "embedding": embedding.tolist(),
                "last_seen": time.time(),
                "seen_count": 1
            }

            self.save()

            return new_id, True

    def update_seen(self, person_id):
        with self.lock:
            self.reset_if_new_day()
            key = str(person_id)

            if key in self.people:
                self.people[key]["last_seen"] = time.time()

    def get_count(self):
        with self.lock:
            self.reset_if_new_day()
            return len(self.people)


def get_person_embedding(embed_model, frame, box):
    x1, y1, x2, y2 = map(int, box)

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    if crop.shape[0] < 40 or crop.shape[1] < 20:
        return None

    try:
        features = embed_model.embed(crop, verbose=False)

        if features is None or len(features) == 0:
            return None

        return features[0].cpu().numpy().flatten()

    except Exception as error:
        print("Embedding error:", error)
        return None


def run_source(source_info, identity_manager):

    source_name = source_info["name"]
    source = source_info["source"]
    is_live = source_info["is_live"]

    print(f"\nStarting source: {source_name}")

    model = YOLO(MODEL_PATH)
    embed_model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Could not open source: {source_name}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    writer = None

    if SAVE_VIDEO:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{source_name}_output.mp4"
        )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

    track_to_global = {}
    track_hits = {}
    frame_number = 0
    last_day = datetime.now().strftime("%Y-%m-%d")

    while True:

        ret, frame = cap.read()

        if not ret:

            if not is_live:
                break

            print(f"Reconnecting: {source_name}")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                continue

            model = YOLO(MODEL_PATH)
            track_to_global.clear()
            track_hits.clear()
            continue

        frame_number += 1

        today = datetime.now().strftime("%Y-%m-%d")

        if today != last_day:
            track_to_global.clear()
            track_hits.clear()
            last_day = today

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            active_global_ids = {
                track_to_global[tid]
                for tid in track_ids
                if tid in track_to_global
            }

            for box, track_id in zip(boxes, track_ids):

                track_hits[track_id] = track_hits.get(track_id, 0) + 1

                if track_hits[track_id] < MIN_HITS:
                    continue

                should_extract = (
                    track_id not in track_to_global
                    or frame_number % EMBED_INTERVAL == 0
                )

                if should_extract:

                    embedding = get_person_embedding(
                        embed_model,
                        frame,
                        box
                    )

                    if embedding is not None:

                        if track_id not in track_to_global:

                            global_id, is_new = identity_manager.identify(
                                embedding,
                                forbidden_ids=active_global_ids
                            )

                            if global_id is not None:
                                track_to_global[track_id] = global_id
                                active_global_ids.add(global_id)

                                if is_new:
                                    print(f"New person: ID {global_id}")

                        else:
                            global_id = track_to_global[track_id]
                            identity_manager.update_seen(global_id)

                if track_id in track_to_global:

                    global_id = track_to_global[track_id]
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Person ID: {global_id}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2
                    )

        total_people = identity_manager.get_count()

        cv2.rectangle(
            frame,
            (10, 10),
            (430, 75),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            f"Unique People Today: {total_people}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if SHOW_WINDOWS:
            cv2.imshow(source_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(frame)

    cap.release()

    if writer is not None:
        writer.release()

    if SHOW_WINDOWS:
        cv2.destroyWindow(source_name)

    print(f"Finished source: {source_name}")


def main():

    identity_manager = GlobalIdentityManager()

    if not SOURCES:
        print("No camera or video sources configured.")
        return

    threads = []

    for source_info in SOURCES:

        thread = threading.Thread(
            target=run_source,
            args=(source_info, identity_manager),
            daemon=False
        )

        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("\n================================")
    print(
        "TOTAL UNIQUE PEOPLE TODAY:",
        identity_manager.get_count()
    )
    print("Program finished!")
    print("================================")


if __name__ == "__main__":
    main()