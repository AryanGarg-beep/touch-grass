# record_test.py - 5-second mic test for the Pico 2 W field recorder (MicroPython).
#
# Wiring (tonight's build): mic + -> GP26 and, through 2.2k, to 3V3 OUT. Mic - -> AGND.
# Button between GP16 and GND (optional).
#
# Run from your laptop:   mpremote run record_test.py
# Then copy the file:     mpremote cp :test.wav .

import array
import struct
import time

import machine

RATE = 8000          # samples per second (phone-call quality, fine for Whisper)
SECONDS = 5

adc = machine.ADC(26)                                   # GP26
button = machine.Pin(16, machine.Pin.IN, machine.Pin.PULL_UP)  # GP16, pressed = 0
USE_LED = True    # onboard LED: blinks while the recorder is on, steady while recording.
                  # It lives on the Wi-Fi chip; if USB starts dropping again, set this to False.


class _NoLed:
    def on(self): pass
    def off(self): pass
    def toggle(self): pass


led = machine.Pin("LED", machine.Pin.OUT) if USE_LED else _NoLed()
_blinker = machine.Timer()


def blink_on(period_ms=400):
    """Blink in the background: the recorder is on and waiting."""
    _blinker.init(period=period_ms, mode=machine.Timer.PERIODIC, callback=lambda t: led.toggle())


def blink_off():
    _blinker.deinit()
    led.off()


def volts(raw):
    return raw * 3.3 / 65535


def wiring_check():
    total = 0
    for _ in range(2000):
        total += adc.read_u16()
    v = volts(total / 2000)
    print("GP26 resting voltage: {:.2f} V".format(v))
    if v < 0.3:
        print("  -> Too low. GP26 is probably touching ground, or the mic + leg isn't on GP26.")
    elif v > 3.1:
        print("  -> Too high. The mic isn't drawing current: check the mic - leg goes to AGND,")
        print("     and that the mic isn't in backwards.")
    else:
        print("  -> Looks right. The mic is powered and connected.")


def wait_for_button(timeout_s=10):
    print("Press the button to record (or wait {} s and it starts anyway)...".format(timeout_s))
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < timeout_s * 1000:
        if button.value() == 0:
            time.sleep_ms(30)  # let the press settle
            return
        time.sleep_ms(20)


def record(n):
    """Fill n samples. Each sample averages as many ADC reads as fit in its slot,
    which lowers the noise of the bare mic."""
    buf = array.array("H", bytes(2 * n))
    period = 1_000_000 // RATE
    read = adc.read_u16
    ticks = time.ticks_us
    diff = time.ticks_diff
    add = time.ticks_add
    reads = 0
    late = 0
    nxt = add(ticks(), period)
    for i in range(n):
        s = 0
        c = 0
        while diff(nxt, ticks()) > 0:
            s += read()
            c += 1
        if c:
            buf[i] = s // c
            reads += c
        else:
            buf[i] = read()
            late += 1
        nxt = add(nxt, period)
    return buf, reads / n, late


def write_wav(path, buf, mean, gain):
    n = len(buf)
    data_bytes = n * 2
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", data_bytes))
        chunk = 1024
        out = array.array("h", bytes(2 * chunk))
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            for j in range(m):
                x = int((buf[start + j] - mean) * gain)
                out[j] = 32767 if x > 32767 else (-32768 if x < -32768 else x)
            f.write(memoryview(out)[:m])


def main():
    print("Script started.")
    blink_on()
    wiring_check()
    wait_for_button()

    print("Recording {} s - talk normally, a hand's width from the mic.".format(SECONDS))
    blink_off()   # stop the blink timer so it can't disturb the sample timing
    led.on()      # steady light = recording
    buf, avg_reads, late = record(RATE * SECONDS)
    led.off()
    blink_on()    # back to blinking: on and waiting

    n = len(buf)
    mean = sum(buf) / n
    peak = max(max(buf) - mean, mean - min(buf))
    p2p_mv = volts(max(buf) - min(buf)) * 1000
    gain = 30000 / peak if peak else 1.0

    write_wav("test.wav", buf, mean, gain)

    print("Saved test.wav")
    print("  signal swing: {:.0f} mV peak-to-peak".format(p2p_mv))
    print("  ADC reads averaged per sample: {:.1f}{}".format(
        avg_reads, "" if late == 0 else "  ({} samples late)".format(late)))
    if p2p_mv < 20:
        print("  -> Very quiet. Speak closer; if it's still faint, the transistor upgrade will fix it.")
    print("Copy it to your laptop with:  mpremote cp :test.wav .")


main()