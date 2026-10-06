"""All drawing code. One overlay blend per frame (the old code blended per object)."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(0, 255, 0), (255, 160, 0), (0, 200, 255), (255, 0, 200),
           (140, 255, 0), (255, 255, 0), (0, 128, 255), (200, 0, 255)]


def put_text(frame, text, org, scale=0.5, color=(255, 255, 255), thickness=1, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x = max(0, min(org[0], frame.shape[1] - w - 4))
    y = max(h + 4, min(org[1], frame.shape[0] - 4))
    cv2.rectangle(frame, (x - 2, y - h - 4), (x + w + 2, y + base - 2), bg, -1)
    cv2.putText(frame, text, (x, y - 2), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_floor_line(frame, floor_y):
    cv2.line(frame, (0, floor_y), (frame.shape[1], floor_y), (0, 0, 255), 2)
    put_text(frame, "Floor", (10, floor_y - 10), 0.6, (0, 0, 255))


def draw_detections(frame, detections, floor_y, show_masks=True, alpha=0.35, draw_links=True):
    colors = [PALETTE[d.label_id % len(PALETTE)] for d in detections]

    if show_masks:                                   # single blend for all masks
        overlay = frame.copy()
        for d, c in zip(detections, colors):
            if d.contour is not None:
                cv2.fillPoly(overlay, [d.contour.reshape(-1, 1, 2)], c)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    for d, c in zip(detections, colors):
        x1, y1, x2, y2 = d.bbox
        if d.contour is not None:
            cv2.drawContours(frame, [d.contour.reshape(-1, 1, 2)], -1, c, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 1)

        label = f"{d.name} #{d.label_id} {d.confidence:.2f} | {d.floor_gap_cm:.1f} cm"
        put_text(frame, label, (x1, y1 - 6), 0.45, (255, 255, 255), 1, bg=(40, 40, 40))

        if draw_links and y2 < floor_y:              # line from object bottom to floor
            cx = (x1 + x2) // 2
            cv2.line(frame, (cx, y2), (cx, floor_y), (255, 0, 0), 2)
            cv2.circle(frame, (cx, y2), 4, (0, 0, 255), -1)
            cv2.circle(frame, (cx, floor_y), 4, (255, 0, 255), -1)


def draw_hud(frame, targets, counts, meter):
    h, w = frame.shape[:2]
    put_text(frame, "Searching: " + ", ".join(targets), (10, 22), 0.55)
    put_text(frame, "Count: " + " | ".join(f"{t}: {counts.get(t, 0)}" for t in targets),
             (10, 46), 0.6, (0, 255, 255))
    missing = [t for t in targets if counts.get(t, 0) == 0]
    if missing:
        put_text(frame, "NOT FOUND: " + ", ".join(missing), (10, 70), 0.6, (0, 0, 255))

    put_text(frame, f"FPS: {meter.realtime_fps:.1f} | Avg: {meter.average_fps:.1f} | "
                    f"Detect: {meter.latency_ms:.0f} ms", (10, h - 12), 0.55, (255, 0, 255))
    put_text(frame, "Q quit | S screenshot | M masks", (w - 250, h - 12), 0.45)
