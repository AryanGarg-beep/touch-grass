# Field Notes

The laptop side of the pocket recorder. Import voice notes from the Pico (or your phone),
listen to them, transcribe them with Whisper, and let Gemma write the journal entry for each
outing plus a weekly recap. Everything runs on this laptop.

## Setup (once)

```bash
cd ~/touch_grass/fieldnotes
source ../venv/bin/activate          # the venv that already has mpremote
pip install faster-whisper            # Whisper; also brings numpy (used to remove the whine)

sudo pacman -S ollama ffmpeg          # ffmpeg is only needed for phone formats like .m4a
sudo systemctl enable --now ollama    # or run `ollama serve` in a spare terminal
ollama pull gemma3:4b                 # ~3.3 GB; on a low-RAM laptop use gemma3:1b
```

The first transcription downloads the Whisper model (about 150 MB), so be online that one time.

## Run

```bash
python app.py
```

It opens as its own window (Chromium/Chrome/Brave) or a browser tab. To get a launcher icon:

```bash
./install_launcher.sh
```

**Quit** in the top corner stops it.

## Use

1. Plug the recorder in and press **Import from recorder**. It copies the notes off, clears the
   recorder (untick the box to keep them), and restarts it.
2. It then transcribes the notes and asks Gemma for the entry by itself (switch that off in
   Settings if you'd rather press the buttons).
3. **Write this week's recap** summarises the past 7 days and checks the goals you said out loud
   ("goal: four walks this week").
4. Click any transcript to fix a word Whisper misheard; then **Rewrite entry**.

Everything lives in `data/walks/<outing>/`: the cleaned audio, the original recordings in `raw/`,
`walk.json`, and `journal.md`.

## The recorder

`recorder/main.py` is the Pico firmware. Install it with
`mpremote connect /dev/ttyACM0 cp recorder/main.py :main.py`, then replug the Pico.
