"""
SG90 360度连续旋转舵机控制

360度舵机通过 PWM 脉宽控制旋转方向和速度:
- 脉宽 ~1.5ms: 停止
- 脉宽 >1.5ms: 正转（开启猫粮）
- 脉宽 <1.5ms: 反转（关闭猫粮）
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SERVO_GPIO_PIN, SERVO_OPEN_DURATION, SERVO_CLOSE_DURATION, COOLDOWN_SECONDS

try:
    from gpiozero import Servo
    from gpiozero.pins.pigpio import PiGPIOFactory
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[舵机] gpiozero 未安装，舵机控制将以模拟模式运行")


class ServoController:
    def __init__(self):
        self._last_trigger_time = 0

        if GPIO_AVAILABLE:
            # 使用 pigpio 工厂获取更精确的 PWM（需要运行 sudo pigpiod）
            try:
                factory = PiGPIOFactory()
                self.servo = Servo(
                    SERVO_GPIO_PIN,
                    initial_value=None,            # 初始化时不发送 PWM，避免误转
                    pin_factory=factory,
                    min_pulse_width=0.5 / 1000,   # 0.5ms
                    max_pulse_width=2.5 / 1000,    # 2.5ms
                )
                print(f"[舵机] 已初始化 GPIO{SERVO_GPIO_PIN} (pigpio)")
            except Exception:
                # pigpio 不可用时退回软件 PWM
                self.servo = Servo(
                    SERVO_GPIO_PIN,
                    initial_value=None,            # 初始化时不发送 PWM
                    min_pulse_width=0.5 / 1000,
                    max_pulse_width=2.5 / 1000,
                )
                print(f"[舵机] 已初始化 GPIO{SERVO_GPIO_PIN} (软件PWM)")
        else:
            self.servo = None
            print("[舵机] 模拟模式")

    @property
    def is_cooling_down(self) -> bool:
        """是否在冷却期内"""
        return (time.time() - self._last_trigger_time) < COOLDOWN_SECONDS

    @property
    def cooldown_remaining(self) -> float:
        """剩余冷却时间（秒）"""
        remaining = COOLDOWN_SECONDS - (time.time() - self._last_trigger_time)
        return max(0, remaining)

    def open_food(self):
        """正转 — 打开猫粮"""
        print(f"[舵机] 正转 {SERVO_OPEN_DURATION}s — 打开猫粮")
        if self.servo:
            self.servo.max()  # 正转最大速度
        time.sleep(SERVO_OPEN_DURATION)
        self.stop()

    def close_food(self):
        """反转 — 关闭猫粮"""
        print(f"[舵机] 反转 {SERVO_CLOSE_DURATION}s — 关闭猫粮")
        if self.servo:
            self.servo.min()  # 反转最大速度
        time.sleep(SERVO_CLOSE_DURATION)
        self.stop()

    def stop(self):
        """停止旋转 — 切断 PWM 信号（比发送 mid 信号更可靠）"""
        if self.servo:
            self.servo.value = None

    def trigger_feed(self):
        """执行一次喂食：开→等待→关，并记录冷却时间"""
        if self.is_cooling_down:
            print(f"[舵机] 冷却中，剩余 {self.cooldown_remaining:.0f}s")
            return False

        self.open_food()
        time.sleep(1)  # 猫粮落下时间
        self.close_food()
        self._last_trigger_time = time.time()
        print(f"[舵机] 喂食完成，冷却 {COOLDOWN_SECONDS}s")
        return True

    def cleanup(self):
        """释放 GPIO 资源"""
        if self.servo:
            self.servo.value = None
            self.servo.close()
            print("[舵机] GPIO 已释放")


# 独立运行时执行测试
if __name__ == "__main__":
    print("舵机测试")
    print("=" * 30)
    ctrl = ServoController()

    try:
        input("按回车测试正转 (打开)...")
        ctrl.open_food()
        input("按回车测试反转 (关闭)...")
        ctrl.close_food()
        input("按回车测试完整喂食流程...")
        ctrl.trigger_feed()
        print("测试完成")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        ctrl.cleanup()
