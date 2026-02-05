#!/usr/bin/env python3
"""
Drone Demo - Autonomous object tracking with DJI Tello (ID Locking + Height Offset).

此版本特性：
1. 自动锁定并跟随特定 ID，不随意切换目标。
2. 十字准星下移，指示无人机将维持更高的高度（+30cm）。
3. 按 'r' 键重置锁定。
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, DroneConfig, get_drone_config
from src.detector import ObjectDetector
from src.drone_controller import DroneController, DroneState, MockDroneController
from src.tracker import ObjectTracker  # 使用多目标追踪器
from src.utils import (
    FPSCounter,
    draw_bbox,
    draw_crosshair,
    draw_info_panel,
    draw_trajectory,
    draw_vector,
)


class DroneDemo:
    """Drone tracking demo application."""

    def __init__(
        self, config: Config, drone_config: DroneConfig, use_mock: bool = False
    ):
        self.config = config
        self.drone_config = drone_config

        # Initialize components
        print("Initializing detector...")
        self.detector = ObjectDetector(config)
        print(f"Loaded {config.model_name} on {config.device}")

        # 使用 ObjectTracker 并增加锁定变量
        self.tracker = ObjectTracker(config)
        self.locked_target_id = None
        self.fps_counter = FPSCounter()

        # Initialize drone controller
        if use_mock:
            print("Using mock drone controller (no hardware)")
            self.drone = MockDroneController(config, drone_config)
            self.use_mock = True
        else:
            print("Using real drone controller")
            self.drone = DroneController(config, drone_config)
            self.use_mock = False

        # State
        self.running = False
        self.show_hud = True
        self.show_fps = True
        self.show_telemetry = True
        self.recording = False
        self.video_writer = None

        # Manual control state
        self.manual_speed = 30

        self._print_controls()

    def _print_controls(self) -> None:
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("CONTROLS")
        print("=" * 60)
        print("Flight:")
        print("  TAB       - Takeoff")
        print("  BACKSPACE - Land")
        print("  ESC       - Emergency stop")
        print("  SPACE     - Toggle tracking mode")
        print("\nManual control (tracking off):")
        print("  w/s       - Forward/Backward")
        print("  a/d       - Left/Right")
        print("  UP/DOWN   - Ascend/Descend")
        print("  LEFT/RIGHT- Rotate Left/Right")
        print("\nDisplay:")
        print("  h - Toggle HUD")
        print("  f - Toggle FPS")
        print("  t - Toggle telemetry")
        print("  r - Reset Lock (Find new target)")
        print("  v - Record video")
        print("  c - Take photo")
        print("  q - Quit (will land first)")
        print("=" * 60)
        print()

    def start(self) -> None:
        """Start the demo."""
        # Connect to drone
        if not self.drone.connect():
            print("Failed to connect to drone")
            return

        print("\nDrone connected! Ready to fly.")
        print("Press TAB to takeoff when ready.")

        self.running = True
        self.run_loop()

    def run_loop(self) -> None:
        """Main processing loop."""
        while self.running:
            # Get frame
            if self.use_mock:
                # For mock, use webcam
                if not hasattr(self, "mock_cap"):
                    self.mock_cap = cv2.VideoCapture(0)
                ret, frame = self.mock_cap.read()
                if not ret:
                    continue
            else:
                frame = self.drone.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

            # Update FPS
            self.fps_counter.update()

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display
            cv2.imshow("Drone Tracking Demo", processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_keypress(key)

        self.cleanup()

    def process_frame(self, frame):
        """Process a single frame with ID locking and height offset viz."""
        display_frame = frame.copy()

        # Detect and track
        if self.drone.is_flying():
            detections = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detections)
            
            target = None
            
            # ID Locking Logic
            if self.locked_target_id is not None:
                if self.locked_target_id in tracked_objects:
                    target = tracked_objects[self.locked_target_id]
                else:
                    self.locked_target_id = None
            
            # Find new target if none locked
            if target is None and len(tracked_objects) > 0:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                best_obj = min(tracked_objects.values(), 
                               key=lambda o: (o.center[0]-center[0])**2 + (o.center[1]-center[1])**2)
                
                self.locked_target_id = best_obj.id
                target = best_obj
                print(f"Locked on new target ID: {target.id}")

            # Autonomous tracking
            if self.drone.tracking_enabled:
                self.drone.track_target(target)

            # Visualize
            for obj in tracked_objects.values():
                is_locked = (obj.id == self.locked_target_id)
                color = (0, 255, 0) if is_locked and self.drone.tracking_enabled else (0, 0, 255)
                if is_locked and not self.drone.tracking_enabled:
                    color = (255, 165, 0) # Orange for locked but manual
                
                label = f"{obj.class_name} ID:{obj.id}"
                display_frame = draw_bbox(
                    display_frame,
                    obj.bbox,
                    label,
                    obj.confidence,
                    color,
                    thickness=3 if is_locked else 1,
                )

                if is_locked:
                    if len(obj.centers) > 1:
                        display_frame = draw_trajectory(
                            display_frame, list(obj.centers), (0, 255, 255), thickness=2
                        )

                    # Draw vector to Offset Center
                    if self.drone.tracking_enabled:
                        h, w = frame.shape[:2]
                        # [修改] 向量指向新的偏移中心
                        target_center_y = (h // 2) + self.config.height_offset
                        offset_center = (w // 2, target_center_y)
                        
                        display_frame = draw_vector(
                            display_frame,
                            obj.center,
                            offset_center,
                            (255, 0, 255),
                            thickness=2,
                        )

        # [修改] 绘制偏移后的十字准星
        h, w = frame.shape[:2]
        crosshair_y = (h // 2) + self.config.height_offset
        display_frame = draw_crosshair(display_frame, center=(w // 2, crosshair_y), color=(0, 0, 255), size=30)

        # Draw HUD
        if self.show_hud:
            display_frame = self.draw_hud(display_frame)

        # Record if enabled
        if self.recording and self.video_writer:
            self.video_writer.write(display_frame)

        return display_frame

    def draw_hud(self, frame):
        """Draw heads-up display."""
        info = {}

        # FPS
        if self.show_fps:
            info["FPS"] = f"{self.fps_counter.get_fps():.1f}"

        # Drone state
        info["State"] = self.drone.state.value.upper()

        # Tracking state
        if self.drone.is_flying():
            tracking_status = "ACTIVE" if self.drone.tracking_enabled else "MANUAL"
            info["Mode"] = tracking_status

        # Target info
        if self.locked_target_id is not None and self.tracker.objects:
             obj = self.tracker.get_object(self.locked_target_id)
             if obj:
                info["Target"] = f"ID:{obj.id}"
                info["Conf"] = f"{obj.confidence:.2f}"
             else:
                info["Target"] = "Lost"
        else:
            info["Target"] = "Scanning"

        # Telemetry
        if self.show_telemetry and not self.use_mock:
            telemetry = self.drone.get_telemetry()
            info["Battery"] = f"{telemetry.get('battery', 0)}%"
            info["Height"] = f"{telemetry.get('height', 0)}cm"
            info["Temp"] = f"{telemetry.get('temperature', 0)}C"

        # Recording indicator
        if self.recording:
            info["REC"] = "●"

        # Draw panel
        frame = draw_info_panel(
            frame,
            info,
            position="top-left",
            bg_color=(0, 0, 0),
            text_color=(0, 255, 0) if self.drone.tracking_enabled else (255, 165, 0),
            alpha=0.7,
        )

        return frame

    def handle_keypress(self, key: int) -> None:
        """Handle keyboard input."""
        # Flight controls
        if key == 9:  # TAB
            if self.drone.state == DroneState.CONNECTED:
                print("Taking off...")
                self.drone.takeoff()

        elif key == 8:  # BACKSPACE
            if self.drone.is_flying():
                print("Landing...")
                self.drone.land()

        elif key == 27:  # ESC
            print("EMERGENCY STOP!")
            self.drone.emergency_stop()
            self.running = False

        elif key == ord(" "):  # SPACE
            if self.drone.is_flying():
                if self.drone.tracking_enabled:
                    self.drone.disable_tracking()
                    print("Tracking disabled - manual control active")
                else:
                    self.drone.enable_tracking()
                    print("Tracking enabled - autonomous mode")

        # Manual controls (only when not tracking)
        elif key == ord("w") and not self.drone.tracking_enabled:
            self.drone.manual_control(fb=self.manual_speed)

        elif key == ord("s") and not self.drone.tracking_enabled:
            self.drone.manual_control(fb=-self.manual_speed)

        elif key == ord("a") and not self.drone.tracking_enabled:
            self.drone.manual_control(lr=-self.manual_speed)

        elif key == ord("d") and not self.drone.tracking_enabled:
            self.drone.manual_control(lr=self.manual_speed)

        elif key == 82 and not self.drone.tracking_enabled:  # UP arrow
            self.drone.manual_control(ud=self.manual_speed)

        elif key == 84 and not self.drone.tracking_enabled:  # DOWN arrow
            self.drone.manual_control(ud=-self.manual_speed)

        elif key == 81 and not self.drone.tracking_enabled:  # LEFT arrow
            self.drone.manual_control(yaw=-self.manual_speed)

        elif key == 83 and not self.drone.tracking_enabled:  # RIGHT arrow
            self.drone.manual_control(yaw=self.manual_speed)

        # Display controls
        elif key == ord("h"):
            self.show_hud = not self.show_hud
            print(f"HUD {'shown' if self.show_hud else 'hidden'}")

        elif key == ord("f"):
            self.show_fps = not self.show_fps
            print(f"FPS display {'shown' if self.show_fps else 'hidden'}")

        elif key == ord("t"):
            self.show_telemetry = not self.show_telemetry
            print(f"Telemetry {'shown' if self.show_telemetry else 'hidden'}")

        elif key == ord("r"):
            print("Resetting tracker and lock...")
            self.tracker.reset()
            self.locked_target_id = None

        elif key == ord("v"):
            self.toggle_recording()

        elif key == ord("c"):
            self.take_photo()

        elif key == ord("q"):
            print("Quitting...")
            if self.drone.is_flying():
                print("Landing before quit...")
                self.drone.land()
            self.running = False

    def toggle_recording(self) -> None:
        """Toggle video recording."""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                self.config.fps,
                (self.config.frame_width, self.config.frame_height),
            )

            self.recording = True
            print(f"Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.recording = False
            print("Recording stopped")

    def take_photo(self) -> None:
        """Take a photo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"

        frame = self.drone.get_frame()
        if frame is not None:
            cv2.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
        else:
            print("Failed to capture photo")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.recording and self.video_writer:
            self.video_writer.release()

        if hasattr(self, "mock_cap"):
            self.mock_cap.release()

        self.drone.disconnect()
        cv2.destroyAllWindows()
        print("Demo stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Drone demo for autonomous object tracking (ID Lock Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s",
        help="YOLO model to use (default: yolov8s)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Detection confidence threshold (default: 0.6)",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Target classes to detect (e.g., person ball)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run model on (default: auto)",
    )

    parser.add_argument(
        "--speed", type=int, default=50, help="Drone movement speed 0-100 (default: 50)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock drone (for testing without hardware)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configurations
    config, drone_config = get_drone_config()
    config.model_name = args.model
    config.confidence_threshold = args.confidence
    config.drone_speed = args.speed

    if args.classes:
        config.target_classes = args.classes

    if args.device:
        config.device = args.device

    # Safety check
    if not args.mock:
        print("\n" + "!" * 60)
        print("SAFETY WARNING")
        print("!" * 60)
        print("You are about to fly a real drone.")
        print("- Ensure you are in an open area")
        print("- Keep away from people and obstacles")
        print("- Monitor battery level")
        print("- Be ready to emergency stop (ESC key)")
        print("!" * 60)

        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted")
            return

    # Create and start demo
    demo = DroneDemo(config, drone_config, use_mock=args.mock)

    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        demo.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        demo.cleanup()


if __name__ == "__main__":
    main()