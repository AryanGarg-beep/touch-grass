# main.py - Pico 2 W pocket voice recorder (MicroPython).
# Runs by itself whenever the Pico gets power (USB, power bank).
#
#   LED slow blink   ready, waiting for the button
#   LED steady       recording
#   LED fast blink   saving the clip
#   LED flicker      problem: storage full, or the mic reads wrong (see printed message)
#
# Press the button to start a note, press again to stop. A note also stops by itself
# after MAX_CLIP_S seconds. Each note is saved as clips/clip_0001.wav, clip_0002.wav, ...
# with one line per note in clips/index.csv (boot number, seconds since power-on, length).
#
# Install:     mpremote cp main.py :main.py   then unplug and replug (or: mpremote reset)
# Copy notes:  mpremote cp -r :clips .
# Clear notes: mpremote exec "import os; [os.remove('clips/'+f) for f in os.listdir('clips')]"

import os
import sys
import time

import machine

RATE = 8000            # samples per second; speech-quality and fine for Whisper
MAX_CLIP_S = 30        # longest single note (RAM holds one note at a time)
ADC_PIN = 26           # GP26: amplifier output (transistor collector)
BUTTON_PIN = 16        # GP16: button to GND
USE_LED = True         # set False if the onboard LED (on the Wi-Fi chip) ever upsets USB
CLIP_DIR = "clips"

adc = machine.ADC(ADC_PIN)
button = machine.Pin(BUTTON_PIN, machine.Pin.IN, machine.Pin.PULL_UP)


class _NoLed:
    def on(self): pass
    def off(self): pass
    def toggle(self): pass


led = machine.Pin("LED", machine.Pin.OUT) if USE_LED else _NoLed()
_timer = machine.Timer()


def led_mode(mode):
    _timer.deinit()
    if mode == "rec":
        led.on()
    elif mode == "off":
        led.off()
    else:
        period = {"idle": 500, "save": 100, "error": 50}[mode]
        _timer.init(period=period, mode=machine.Timer.PERIODIC, callback=lambda t: led.toggle())


# ---- mu-law: 8 bits per sample with the dynamic range of ~14-bit audio ------------------

def _ulaw_table():
    """16384-entry lookup: index = 14-bit signed sample + 8192 -> mu-law byte (G.711)."""
    table = bytearray(16384)
    for i in range(16384):
        pcm = (i - 8192) << 2
        sign = 0x80 if pcm < 0 else 0
        if pcm < 0:
            pcm = -pcm
        if pcm > 32635:
            pcm = 32635
        pcm += 0x84
        exp = 7
        mask = 0x4000
        while exp > 0 and not (pcm & mask):
            exp -= 1
            mask >>= 1
        mant = (pcm >> (exp + 3)) & 0x0F
        table[i] = ~(sign | (exp << 4) | mant) & 0xFF
    return table


# ---- helpers ------------------------------------------------------------------------------

def pressed():
    if button.value() == 0:
        time.sleep_ms(30)
        return button.value() == 0
    return False


def wait_release():
    while button.value() == 0:
        time.sleep_ms(10)
    time.sleep_ms(30)


def measure_dc(n=4000):
    total = 0
    for _ in range(n):
        total += adc.read_u16()
    return total // n


def free_samples():
    st = os.statvfs("/")
    free = st[0] * st[3] - 16384          # keep 16 KB spare for the index and headers
    return max(0, free - 64)               # one byte per sample


def next_boot_number():
    n = 0
    try:
        with open("boot_count.txt") as f:
            n = int(f.read().strip() or 0)
    except (OSError, ValueError):
        pass
    n += 1
    with open("boot_count.txt", "w") as f:
        f.write(str(n))
    return n


def next_clip_name():
    try:
        os.mkdir(CLIP_DIR)
    except OSError:
        pass
    last = 0
    for name in os.listdir(CLIP_DIR):
        if name.startswith("clip_") and name.endswith(".wav"):
            try:
                last = max(last, int(name[5:9]))
            except ValueError:
                pass
    return "{}/clip_{:04d}.wav".format(CLIP_DIR, last + 1)


def alloc_buffer():
    for seconds in (MAX_CLIP_S, 20, 10, 5):
        try:
            return bytearray(seconds * RATE)
        except MemoryError:
            pass
    raise MemoryError("not enough RAM for a recording buffer")


# ---- recording ------------------------------------------------------------------------------

def record_clip(buf, max_n, dc, table):
    """Sample at RATE until the button is pressed again or max_n samples.
    Each sample averages every ADC read that fits in its time slot (less noise)."""
    read = adc.read_u16
    btn = button.value
    ticks = time.ticks_us
    diff = time.ticks_diff
    add = time.ticks_add
    period = 1_000_000 // RATE
    min_n = RATE // 2                      # ignore stop presses in the first half second
    released = False
    n = 0
    nxt = add(ticks(), period)
    while n < max_n:
        s = 0
        c = 0
        while diff(nxt, ticks()) > 0:
            s += read()
            c += 1
        v = (s // c) if c else read()
        i = ((v - dc) >> 2) + 8192
        if i < 0:
            i = 0
        elif i > 16383:
            i = 16383
        buf[n] = table[i]
        n += 1
        nxt = add(nxt, period)
        if n & 255 == 0:                   # check the button every 32 ms
            if btn():
                released = True
            elif released and n > min_n:
                break
    return n


def save_clip(buf, n, boot, start_s):
    name = next_clip_name()
    with open(name, "wb") as f:
        # WAV, mu-law (format 7), mono, 8-bit
        f.write(b"RIFF")
        f.write((4 + 26 + 12 + 8 + n).to_bytes(4, "little"))
        f.write(b"WAVEfmt ")
        f.write((18).to_bytes(4, "little"))
        f.write((7).to_bytes(2, "little") + (1).to_bytes(2, "little"))
        f.write(RATE.to_bytes(4, "little") + RATE.to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little") + (8).to_bytes(2, "little") + (0).to_bytes(2, "little"))
        f.write(b"fact" + (4).to_bytes(4, "little") + n.to_bytes(4, "little"))
        f.write(b"data" + n.to_bytes(4, "little"))
        mv = memoryview(buf)
        for start in range(0, n, 4096):
            f.write(mv[start:min(n, start + 4096)])
    with open(CLIP_DIR + "/index.csv", "a") as f:
        f.write("{},{},{},{:.1f}\n".format(name.split("/")[-1], boot, start_s, n / RATE))
    return name


# ---- main loop ------------------------------------------------------------------------------

def main():
    led_mode("idle")
    boot = next_boot_number()
    table = _ulaw_table()
    buf = alloc_buffer()
    t0 = time.ticks_ms()

    dc = measure_dc()
    volts = dc * 3.3 / 65535
    print("Recorder ready (boot {}). Mic resting at {:.2f} V.".format(boot, volts))
    print("Room for about {:.1f} min of notes; up to {} s per note.".format(
        free_samples() / RATE / 60, len(buf) // RATE))
    if not 0.3 < volts < 3.0:
        print("Mic reads wrong: check the wiring (expected about 1.3-2.0 V).")
        led_mode("error")
        time.sleep(3)
        led_mode("idle")

    while True:
        if pressed():
            room = min(len(buf), free_samples())
            if room < RATE * 2:
                print("Storage full: copy the clips off and clear them.")
                led_mode("error")
                time.sleep(2)
                wait_release()
                led_mode("idle")
                continue
            dc = measure_dc(2000)
            start_s = time.ticks_diff(time.ticks_ms(), t0) // 1000
            led_mode("rec")
            n = record_clip(buf, room, dc, table)
            led_mode("save")
            name = save_clip(buf, n, boot, start_s)
            print("Saved {} ({:.1f} s)".format(name, n / RATE))
            wait_release()
            led_mode("idle")
        time.sleep_ms(20)


try:
    main()
except KeyboardInterrupt:          # mpremote connecting from the laptop
    led_mode("off")
    raise
except Exception as e:
    sys.print_exception(e)
    led_mode("error")
