#!/usr/bin/env python3
"""
Drone Demo - Autonomous object tracking with DJI Tello (ID Locking Version).

此版本已修改为：
1. 使用 ObjectTracker 进行多目标 ID 记忆。
2. 自动锁定画面中心的目标 ID。
3. 即使目标移动到边缘，只要 ID 未丢失，就持续跟随该 ID，忽略其他目标。
4. 按 'r' 键可以重置锁定，重新寻找中心目标。
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
from src.tracker import ObjectTracker
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

        # Initialize Tracker
        self.tracker = ObjectTracker(config)
        self.locked_target_id = None  # 记录当前锁定的目标 ID

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
        """Process a single frame with ID locking logic."""
        display_frame = frame.copy()

        # Detect and track
        # 注意：如果 drone 未起飞或未进入状态，也可以进行检测以便预览
        if self.drone.is_flying() or self.use_mock or True:
            # 1. 检测 (Detect)
            # 如果是 'yolo' 模式，detector 内部会处理 ID 分配
            detections = self.detector.detect(frame)
            
            # 2. 更新追踪器 (Tracker Update)
            # 如果是 'yolo' 模式，tracker 会直接沿用 detector 给的 ID
            # 如果是 'custom' 模式，tracker 会使用距离算法计算 ID
            tracked_objects = self.tracker.update(detections)
            
            target = None
            
            # 3. ID 锁定逻辑 (ID Locking)
            if self.locked_target_id is not None:
                # 检查之前锁定的 ID 是否依然存在于当前的追踪列表中
                if self.locked_target_id in tracked_objects:
                    target = tracked_objects[self.locked_target_id]
                else:
                    # 目标丢失 (Tracker 认为该 ID 消失了)
                    # 此时不立即重置 locked_target_id，等待它可能重现 (由 Tracker 决定何时彻底删除)
                    # 但在当前帧，我们没有目标
                    pass
                    # 如果需要自动切换目标，可以在这里设置 self.locked_target_id = None
            
            # 4. 如果没有锁定目标 (或目标已彻底跟丢)，寻找新的目标
            # 策略：选择离画面中心最近的物体
            if self.locked_target_id is None or (target is None and self.locked_target_id not in tracked_objects):
                 if len(tracked_objects) > 0:
                    h, w = frame.shape[:2]
                    center = (w // 2, h // 2)
                    
                    # 找离中心最近的
                    best_obj = min(tracked_objects.values(), 
                                   key=lambda o: (o.center[0]-center[0])**2 + (o.center[1]-center[1])**2)
                    
                    self.locked_target_id = best_obj.id
                    target = best_obj
                    print(f"Locked on new target ID: {target.id}")

            # 5. 自主跟随 (只跟随 target)
            # 只有当 target 存在且当前处于 tracking_enabled 模式时才发送指令
            if self.drone.tracking_enabled and target is not None:
                self.drone.track_target(target)
            elif self.drone.tracking_enabled and target is None:
                # 开启了跟随但找不到目标：悬停
                self.drone.hover()

            # 6. 可视化
            for obj in tracked_objects.values():
                is_locked = (obj.id == self.locked_target_id)
                
                # 颜色区分：锁定目标为绿色，其他为橙色(未激活)或红色
                if is_locked:
                    color = (0, 255, 0) if self.drone.tracking_enabled else (0, 139, 139) # 绿/黄
                    thickness = 3
                else:
                    color = (0, 0, 255) # 红
                    thickness = 1
                    
                # 绘制边框
                label = f"{obj.class_name} ID:{obj.id}"
                display_frame = draw_bbox(
                    display_frame,
                    obj.bbox,
                    label,
                    obj.confidence,
                    color,
                    thickness=thickness,
                )

                # 仅为锁定目标绘制轨迹和向量
                if is_locked:
                    if len(obj.centers) > 1:
                        display_frame = draw_trajectory(
                            display_frame, list(obj.centers), color, thickness=2
                        )

                    # Draw tracking vector
                    if self.drone.tracking_enabled:
                        h, w = frame.shape[:2]
                        frame_center = (w // 2, h // 2)
                        display_frame = draw_vector(
                            display_frame,
                            obj.center,
                            frame_center,
                            (255, 0, 255),
                            thickness=2,
                        )

        # Draw frame center crosshair
        display_frame = draw_crosshair(display_frame, color=(0, 0, 255), size=30)

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
        
        # [新增] 显示追踪算法模式
        info["Algo"] = self.config.tracking_method.upper()

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
            info["Bat"] = f"{telemetry.get('battery', 0)}%"
            info["H"] = f"{telemetry.get('height', 0)}cm"

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
        # [修改] 默认使用 YOLO-Worldv2
        default="yolov8s-worldv2",
        help="YOLO model to use (default: yolov8s-worldv2)",
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

    # [新增] 跟踪模式选择参数
    parser.add_argument(
        "--tracking-method",
        type=str,
        default="custom",
        choices=["custom", "yolo"],
        help="Tracking method: 'custom' (simple distance) or 'yolo' (ByteTrack/BoT-SORT)",
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
    
    # [新增] 应用跟踪模式
    config.tracking_method = args.tracking_method

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