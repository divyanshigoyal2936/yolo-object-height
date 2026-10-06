"""Interactive calibration for floor line and cm-per-pixel.

Steps (camera must stay fixed afterwards):
  1. Press SPACE to freeze a frame (or pass --image).
  2. Click the FLOOR LINE (the y level the objects stand on).
  3. Click two ends of a reference length at the same depth (e.g. a ruler lying on the floor).
  4. Press ENTER, then type the real length in cm in the terminal.
Writes calibration.yaml, which overrides floor_y / pixel_to_cm from config.yaml.
"""
import argparse
import math

import cv2
import yaml


def open_capture(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)   # reliable on Windows
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(source))
        return cap
    return cv2.VideoCapture(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--image")
    ap.add_argument("--out", default="calibration.yaml")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    else:
        cap = open_capture(args.source)
        frame, failed = None, 0
        while True:
            ok, live = cap.read()
            if not ok:
                failed += 1
                if failed < 30:
                    continue
                break
            failed = 0
            view = live.copy()
            cv2.putText(view, "SPACE = freeze frame", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("Calibrate", view)
            if cv2.waitKey(1) & 0xFF == 32:
                frame = live.copy()
                break
        cap.release()
    if frame is None:
        raise SystemExit("No frame captured.")

    clicks = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((x, y))

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", on_mouse)
    prompts = ["Click the FLOOR line", "Click reference point A", "Click reference point B",
               "Press ENTER to confirm (R to reset)"]
    while True:
        view = frame.copy()
        if len(clicks) >= 1:
            cv2.line(view, (0, clicks[0][1]), (view.shape[1], clicks[0][1]), (0, 0, 255), 2)
        for pt in clicks[1:]:
            cv2.circle(view, pt, 5, (0, 255, 0), -1)
        if len(clicks) == 3:
            cv2.line(view, clicks[1], clicks[2], (0, 255, 0), 2)
        cv2.putText(view, prompts[len(clicks)], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow("Calibrate", view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            clicks.clear()
        if key in (13, 10) and len(clicks) == 3:
            break
        if key == 27:
            raise SystemExit("Cancelled.")
    cv2.destroyAllWindows()

    pixel_len = math.dist(clicks[1], clicks[2])
    real_cm = float(input("Real length between the two points (cm): "))
    pixel_to_cm = round(real_cm / pixel_len, 5)
    data = {"height": {"floor_y": int(clicks[0][1]), "pixel_to_cm": pixel_to_cm}}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    print(f"Saved {args.out}: {data}")


if __name__ == "__main__":
    main()
