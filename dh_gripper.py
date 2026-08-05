import time

import minimalmodbus
import serial


class DHGripper:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        slave_id=1,
        baudrate=115200,
        timeout=0.2,
    ):
        self.inst = minimalmodbus.Instrument(port, slave_id)

        self.inst.serial.baudrate = baudrate
        self.inst.serial.bytesize = 8
        self.inst.serial.parity = serial.PARITY_NONE
        self.inst.serial.stopbits = 1
        self.inst.serial.timeout = timeout

        self.inst.mode = minimalmodbus.MODE_RTU
        self.inst.clear_buffers_before_each_transaction = True

    def initialize(self, wait_time=1.0):
        self.inst.write_register(
            0x0100,
            1,
            functioncode=6,
        )
        time.sleep(wait_time)

    def set_force(self, force):
        force = max(20, min(100, int(force)))

        self.inst.write_register(
            0x0101,
            force,
            functioncode=6,
        )

    def move(
        self,
        position,
        wait=True,
        timeout=5.0,
        tolerance=5,
    ):
        position = max(0, min(1000, int(position)))

        self.inst.write_register(
            0x0103,
            position,
            functioncode=6,
        )

        if wait:
            return self.wait_until_position(
                target=position,
                timeout=timeout,
                tolerance=tolerance,
            )

        return True

    def open(self, wait=True):
        return self.move(1000, wait=wait)

    def close(self, wait=True):
        return self.move(0, wait=wait)

    def wait_until_position(
        self,
        target,
        timeout=5.0,
        tolerance=5,
        poll_interval=0.05,
    ):
        start_time = time.time()

        while time.time() - start_time < timeout:
            current = self.get_position()

            if abs(current - target) <= tolerance:
                return True

            time.sleep(poll_interval)

        return False

    def get_init_state(self):
        return self.inst.read_register(
            0x0200,
            functioncode=3,
        )

    def get_state(self):
        return self.inst.read_register(
            0x0201,
            functioncode=3,
        )

    def get_position(self):
        return self.inst.read_register(
            0x0202,
            functioncode=3,
        )

    def get_target(self):
        return self.inst.read_register(
            0x0103,
            functioncode=3,
        )

    def get_force(self):
        return self.inst.read_register(
            0x0101,
            functioncode=3,
        )

    def print_status(self):
        print("----------------------------")
        print("Init    :", self.get_init_state())
        print("State   :", self.get_state())
        print("Force   :", self.get_force())
        print("Target  :", self.get_target())
        print("Position:", self.get_position())
        print("----------------------------")

    def close_port(self):
        if self.inst.serial.is_open:
            self.inst.serial.close()
