"""Generates a synthetic crowd video clip for testing and offline development."""

from pathlib import Path
import cv2
import numpy as np


def generate_crowd_video(
    output_path: str | Path = "data/sample_crowd.mp4",
    num_frames: int = 40,
    width: int = 640,
    height: int = 360,
    fps: int = 10,
) -> Path:
    """Creates a synthetic multi-person crowd walking video."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    # Initialize simulated pedestrians (x, y, dx, dy, radius, color)
    np.random.seed(42)
    num_people = 30
    people = []
    for _ in range(num_people):
        x = np.random.uniform(50, width - 50)
        y = np.random.uniform(100, height - 50)
        # People closer to bottom (foreground) are larger
        scale = 0.5 + (y / height) * 0.8
        dx = np.random.uniform(-1.5, 1.5)
        dy = np.random.uniform(0.5, 2.5)  # moving generally forward towards camera
        color = (
            int(np.random.randint(60, 200)),
            int(np.random.randint(60, 200)),
            int(np.random.randint(60, 200)),
        )
        people.append({"x": x, "y": y, "dx": dx, "dy": dy, "scale": scale, "color": color})

    for f in range(num_frames):
        # Draw floor background with perspective lines
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Floor surface
        cv2.rectangle(frame, (0, 70), (width, height), (75, 75, 80), -1)

        # Perspective corridor lines
        cv2.line(frame, (100, 70), (0, height), (50, 50, 55), 2)
        cv2.line(frame, (width - 100, 70), (width, height), (50, 50, 55), 2)

        # Draw pedestrians (heads and shoulders)
        # Sort by y so foreground persons occlude background persons
        people.sort(key=lambda p: p["y"])

        for p in people:
            # Update position
            p["x"] += p["dx"]
            p["y"] += p["dy"]

            # Wrap around boundaries
            if p["y"] > height + 20:
                p["y"] = 80
                p["x"] = np.random.uniform(100, width - 100)
            if p["x"] < 30 or p["x"] > width - 30:
                p["dx"] *= -1

            p["scale"] = 0.5 + (p["y"] / height) * 0.8
            head_r = int(10 * p["scale"])
            torso_w = int(22 * p["scale"])
            torso_h = int(28 * p["scale"])
            cx, cy = int(p["x"]), int(p["y"])

            # Torso / Body
            cv2.ellipse(
                frame,
                (cx, cy + head_r + torso_h // 3),
                (torso_w // 2, torso_h // 2),
                0,
                0,
                360,
                p["color"],
                -1,
            )
            # Head
            head_color = (210, 180, 160)
            cv2.circle(frame, (cx, cy), head_r, head_color, -1)
            # Hair/Cap
            cv2.ellipse(frame, (cx, cy - head_r // 3), (head_r, head_r // 2), 0, 180, 360, (20, 20, 20), -1)

        # Add simulated label watermark as required by project specs
        cv2.putText(
            frame,
            "SYNTHETIC CROWD CLIP [SIMULATED]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"Generated {num_frames} frames crowd video at: {out_file.resolve()}")
    return out_file


if __name__ == "__main__":
    generate_crowd_video()
