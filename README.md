# Field Notes

A pocket voice recorder for going outside with your phone put away, and a laptop app that
turns what you said into a journal.

The recorder is a Raspberry Pi Pico 2 W with a microphone and one button: press to start a
note, press again to stop. Back home, the app copies the notes off, transcribes them with
Whisper, and has Gemma (through Ollama) write a journal entry for each outing and a weekly
recap. Everything runs on your laptop.

## What's here

```
fieldnotes/
  app.py               the laptop app (a small local web server + window)
  ui.html              its interface
  fonts/               bundled fonts (SIL OFL), so the app loads nothing from the internet
  install_launcher.sh  adds Field Notes to your app launcher
  recorder/main.py     the Pico firmware
record_test.py         5-second mic test for the Pico, to check the wiring
```

## The recorder

- Raspberry Pi Pico 2 W running MicroPython
- Electret mic, amplified by one transistor, into **GP26** (ADC0)
- Push button from **GP16** to GND
- Any USB power bank

Check the wiring first:

```bash
mpremote run record_test.py      # records 5 s and prints the mic's resting voltage
mpremote cp :test.wav .
```

Install the firmware, then unplug and replug:

```bash
mpremote cp fieldnotes/recorder/main.py :main.py
```

| LED | Meaning |
|---|---|
| slow blink | ready |
| steady | recording |
| fast blink | saving |
| flicker | storage full, or the mic reads wrong |

Notes are 8 kHz mu-law WAVs (`clips/clip_0001.wav`, ...) with one line each in
`clips/index.csv`: file, boot number, seconds since power-on, length. A note stops by itself
after 30 seconds.

## The app

### Setup (once)

```bash
python -m venv venv && source venv/bin/activate
pip install mpremote faster-whisper     # faster-whisper also brings numpy (whine removal)

# Ollama with a Gemma model (Arch/CachyOS shown)
sudo pacman -S ollama ffmpeg            # ffmpeg only for phone formats like .m4a
sudo systemctl enable --now ollama
ollama pull gemma4:e4b-it-qat           # or gemma4:e2b-it-qat on a low-RAM laptop
```

The first transcription downloads the Whisper model (about 150 MB for `base.en`), so be online
that one time. If the recorder shows "no permission", add yourself to the serial group:
`sudo usermod -aG uucp $USER`, then log out and back in.

**Optional, for the weather banner:** put a [SerpApi](https://serpapi.com) key in
`fieldnotes/.env`. That file is git-ignored.

```
SERPAPI_KEY=your_key_here
```

### Run

```bash
python fieldnotes/app.py                # opens its own window (Chromium/Chrome/Brave) or a tab
./fieldnotes/install_launcher.sh        # optional: a launcher icon
```

**Quit** in the top bar stops it. **?** explains everything in the app.

### Use

- **Import from recorder**: plug the Pico in by USB. Its notes become one outing, get
  transcribed, and Gemma writes the entry. Tick *Clear the recorder afterwards* to empty it.
- **Import a saved recorder folder**: a `clips` folder you copied off the recorder earlier.
  Include its `index.csv` to keep the timings.
- **Add phone recordings**: any audio files; together they become one outing.
- **Fix a transcript**: click it to correct a misheard word, then press **Rewrite entry**.
- **Merge into the outing above**: joins two outings that were really one walk.
- **Write this week's recap**: summarises the past 7 days and checks the goals you said
  out loud ("goal: three walks this week").

Times are minutes after setting out. If the recorder restarted during a walk (a loose power
bank, say), its clock starts again from zero. Notes after a restart are placed after the
previous ones and marked **≈** as estimates.

### "Should you go outside?"

With a SerpApi key set, a banner shows the weather now and for the next 3 hours. Gemma
decides whether it's time to touch grass, whether to go later, or whether to stay in.

- **Location:** in **Settings**, enter an Indian PIN code and pick your area, or type a city.
- **Your limits:** set `max_temp_c` (default 34) and `max_precip_pct` (default 40) in
  `fieldnotes/data/config.json`. Gemma uses them when deciding.

## Your data

Everything lives in `fieldnotes/data/` (git-ignored). Each outing is in `walks/<outing>/`:
the cleaned audio, the original recordings in `raw/`, `walk.json` and `journal.md`.

Nothing leaves the laptop, except:
- the one-time Whisper model download,
- your area name, sent to SerpApi at most once an hour (only if a key is set),
- a PIN code, sent to India Post's public API when you look one up.

## Credits

Fonts: Barlow Semi Condensed and Martian Mono, under the SIL Open Font License 1.1
(see `fieldnotes/fonts/FONTS.txt`).
