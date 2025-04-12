import sys
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import ffmpeg
import torch
import whisper
# import librosa # Librosa might not be strictly needed if ffmpeg handles conversion well
from tqdm import tqdm
import tempfile
import shutil

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar, QMessageBox,
    QComboBox, QStatusBar
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
    ]
)
logger = logging.getLogger(__name__)

# --- Dataclass for Transcript Segments (Slightly Modified) ---
@dataclass
class TranscriptSegment:
    segment_id: int
    start: float
    end: float
    text: str
    # Speaker is removed as basic Whisper doesn't provide it

    def format_timestamp(self, seconds: float) -> str:
        """Formats seconds into HH:MM:SS,ms """
        if seconds < 0: seconds = 0.0 # Ensure non-negative
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}" # Use comma for SRT

    def to_srt_format(self) -> str:
        """Formats the segment in SubRip (.srt) format."""
        start_str = self.format_timestamp(self.start)
        end_str = self.format_timestamp(self.end)
        return f"{self.segment_id}\n{start_str} --> {end_str}\n{self.text}\n"

    def to_simple_format(self) -> str:
        """Formats the segment as [HH:MM:SS,ms] Text."""
        start_str = self.format_timestamp(self.start)
        return f"[{start_str}] {self.text}"

# --- Backend Logic ---

class AudioProcessor:

    def get_ffmpeg_path(self) -> str:
        """Determines the path to ffmpeg executable, prioritizing bundled version."""
        if getattr(sys, 'frozen', False): # Bundled check
            application_path = Path(sys.executable).parent
        else: # Development fallback
            application_path = Path(__file__).parent

        ffmpeg_exe = application_path / "ffmpeg.exe"
        # logger.info(f"Looking for ffmpeg at: {ffmpeg_exe}") # Use logger if configured

        if not ffmpeg_exe.is_file():
            # logger.warning(f"Bundled ffmpeg.exe not found at {ffmpeg_exe}. Trying system PATH.")
            # Decide: either raise error or return None to allow PATH fallback (less self-contained)
            raise RuntimeError(f"Bundled ffmpeg.exe not found at expected location: {ffmpeg_exe}")
            # return None # Alternative: Allow PATH fallback

        # logger.info(f"Using ffmpeg found at: {ffmpeg_exe}")
        return str(ffmpeg_exe)

    def convert_to_wav(self, input_path: str, output_dir: Path) -> str:
        # ... (try block, setup temp_wav_path) ...
        try:
            # ... setup temp_wav_path in system temp dir ...
            temp_wav_filename = f"transcriber_temp_{Path(input_path).stem}_{os.getpid()}.wav"
            temp_wav_path = Path(tempfile.gettempdir()) / temp_wav_filename

            ffmpeg_executable = self.get_ffmpeg_path() # Get bundled path

            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream.audio, str(temp_wav_path), acodec='pcm_s16le', ac=1, ar='16k'
            )

            run_kwargs = {"overwrite_output": True, "capture_stdout": True, "capture_stderr": True}
            if ffmpeg_executable: # IMPORTANT: Use the found path
                run_kwargs["cmd"] = ffmpeg_executable

            ffmpeg.run(stream, **run_kwargs)
            return str(temp_wav_path)
        # ... (except blocks) ...
        except ffmpeg.Error as e:
            error_message = e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error'
            # logger.error(f"FFmpeg error during conversion: {error_message}") # Use logger
            raise RuntimeError(f"FFmpeg conversion error for '{input_path}': {error_message}") from e
        except Exception as e:
            # logger.error(f"Error converting file '{input_path}': {e}", exc_info=True) # Use logger
            raise
    def is_media_file(self, file_path: str) -> bool:
        """Checks if the file extension suggests a common audio/video format."""
        media_extensions = [
            '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', # Video
            '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'         # Audio
        ]
        return Path(file_path).suffix.lower() in media_extensions

    def needs_conversion(self, file_path: str) -> bool:
        """Checks if the file likely needs conversion to WAV for Whisper."""
        # Whisper generally prefers WAV, but can handle others.
        # Let's convert non-WAV files for maximum compatibility.
        return Path(file_path).suffix.lower() != '.wav'

    def convert_to_wav(self, input_path: str, output_dir: Path) -> str:
        """Converts input audio/video file to WAV format in the output directory."""
        try:
            input_path_obj = Path(input_path)
            audio_filename = f"{input_path_obj.stem}_converted.wav"
            # Use a temporary directory for conversion output
            temp_wav_path = Path(tempfile.gettempdir()) / f"transcriber_temp_{os.getpid()}_{audio_filename}"

            logger.info(f"Converting '{input_path}' to WAV format at '{temp_wav_path}'")
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream.audio, # Select only the audio stream
                str(temp_wav_path),
                acodec='pcm_s16le', # Signed 16-bit PCM
                ac=1,             # Mono channel
                ar='16k'          # 16kHz sample rate
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            logger.info(f"Conversion complete: {temp_wav_path}")
            return str(temp_wav_path)

        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else 'Unknown FFmpeg error'
            logger.error(f"FFmpeg error during conversion: {error_message}")
            raise RuntimeError(f"FFmpeg conversion error: {error_message}") from e
        except Exception as e:
            logger.error(f"Error converting file: {e}", exc_info=True)
            raise

# --- Transcription Worker ---
class TranscriptionWorker(QObject):
    """Runs transcription in a separate thread."""
    progress = pyqtSignal(int, str) # Percentage, Status message
    finished = pyqtSignal(list)     # List of TranscriptSegment objects
    error = pyqtSignal(str)         # Error message
    model_loaded = pyqtSignal()     # Signal when model is ready

    def __init__(self, input_filepath: str, model_name: str):
        super().__init__()
        self.input_filepath = input_filepath
        self.model_name = model_name
        self.audio_processor = AudioProcessor()
        self.model = None
        self.temp_wav_path = None # To store path of converted WAV if created


    def load_model(self):
        """Loads the whisper model."""
        try:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self.progress.emit(5, f"Loading model '{self.model_name}'...")
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA available, using GPU.")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU.")
            self.model = whisper.load_model(self.model_name, device=device)
            logger.info("Whisper model loaded.")
            self.model_loaded.emit() # Signal that model is ready
        except Exception as e:
             logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
             self.error.emit(f"Failed to load Whisper model: {e}")


    def run_transcription(self):
        """Performs audio conversion and transcription."""
        if not self.model:
             self.error.emit("Transcription model not loaded.")
             return

        try:
            audio_to_transcribe = self.input_filepath
            self.temp_wav_path = None # Reset potential temp path

            # 1. Convert to WAV if necessary
            if self.audio_processor.needs_conversion(self.input_filepath):
                self.progress.emit(10, "Converting file to WAV...")
                try:
                    # Convert in a standard temp directory
                    self.temp_wav_path = self.audio_processor.convert_to_wav(
                        self.input_filepath, Path(tempfile.gettempdir())
                    )
                    audio_to_transcribe = self.temp_wav_path
                except Exception as e:
                    self.error.emit(f"Audio conversion failed: {e}")
                    return # Stop processing if conversion fails
            else:
                logger.info("Input is WAV, no conversion needed.")
                # Optionally copy to temp dir anyway to avoid issues with special chars/paths
                # self.temp_wav_path = shutil.copy(self.input_filepath, tempfile.gettempdir())
                # audio_to_transcribe = self.temp_wav_path


            # 2. Transcribe
            logger.info(f"Starting transcription for '{audio_to_transcribe}'...")
            self.progress.emit(20, "Starting transcription...")

            # --- Whisper Transcribe ---
            # Note: Whisper's internal progress isn't easily captured for smooth UI updates.
            # We'll just show "Transcribing..." and update when done.
            # For very long files, consider chunking if UI feedback during transcription is critical.
            result = self.model.transcribe(audio_to_transcribe, verbose=False) # verbose=True logs to console
            # --------------------------

            self.progress.emit(90, "Processing results...")
            segments_data = result.get("segments", [])
            segments = []
            for idx, seg_data in enumerate(segments_data):
                start_time = seg_data.get("start", 0.0)
                end_time = seg_data.get("end", start_time)
                text_content = seg_data.get("text", "").strip()

                if not text_content: continue # Skip empty segments

                segment = TranscriptSegment(
                    segment_id=idx + 1, # Simple 1-based index
                    start=start_time,
                    end=end_time,
                    text=text_content
                )
                segments.append(segment)

            logger.info(f"Transcription complete. Found {len(segments)} segments.")
            self.progress.emit(100, "Transcription finished.")
            self.finished.emit(segments)

        except Exception as e:
            logger.error(f"Transcription process failed: {e}", exc_info=True)
            self.error.emit(f"Transcription failed: {e}")
        finally:
             # Clean up temporary WAV file if created
            if self.temp_wav_path and os.path.exists(self.temp_wav_path):
                try:
                    os.remove(self.temp_wav_path)
                    logger.info(f"Cleaned up temporary file: {self.temp_wav_path}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary file {self.temp_wav_path}: {e}")


# --- Main Application Window ---
class TranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio/Video Transcriber")
        self.setGeometry(100, 100, 700, 550) # x, y, width, height

        self.input_filepath = None
        self.transcript_segments: List[TranscriptSegment] = []
        self.transcription_thread = None
        self.worker = None

        # UI Elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # File Selection Row
        self.file_select_layout = QHBoxLayout()
        self.select_button = QPushButton("Select File...")
        self.select_button.clicked.connect(self.select_file)
        self.filepath_label = QLabel("No file selected.")
        self.filepath_label.setWordWrap(True)
        self.file_select_layout.addWidget(self.select_button)
        self.file_select_layout.addWidget(self.filepath_label, 1) # Stretch label

        # Model Selection Row
        self.model_select_layout = QHBoxLayout()
        self.model_label = QLabel("Whisper Model:")
        self.model_combo = QComboBox()
        # Add models (consider adding more like 'small', 'medium', 'large' if needed)
        self.model_combo.addItems(["tiny", "base", "small", "medium"]) # "large" can be very demanding
        self.model_combo.setCurrentText("base") # Default model
        self.model_select_layout.addWidget(self.model_label)
        self.model_select_layout.addWidget(self.model_combo)
        self.model_select_layout.addStretch() # Push combo to the left

        # Transcribe Button
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setEnabled(False) # Disabled until file is selected
        self.transcribe_button.clicked.connect(self.start_transcription)

        # Transcript Display
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setPlaceholderText("Transcript will appear here...")

        # Save Button
        self.save_button = QPushButton("Save Transcript (.srt)")
        self.save_button.setEnabled(False) # Disabled until transcription is done
        self.save_button.clicked.connect(self.save_transcript)

        # Progress Bar & Status Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False) # Hide initially
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add widgets to layout
        self.layout.addLayout(self.file_select_layout)
        self.layout.addLayout(self.model_select_layout)
        self.layout.addWidget(self.transcribe_button)
        self.layout.addWidget(self.transcript_display, 1) # Make text area stretch
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.save_button)


    def select_file(self):
        """Opens a dialog to select an audio or video file."""
        # Define filters for common types
        file_filter = "Media Files (*.mp3 *.wav *.m4a *.aac *.ogg *.flac *.mp4 *.mov *.avi *.mkv *.wmv *.webm);;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Audio or Video File", "", file_filter
        )
        if filepath:
            self.input_filepath = filepath
            self.filepath_label.setText(f"Selected: {os.path.basename(filepath)}")
            self.transcribe_button.setEnabled(True)
            self.save_button.setEnabled(False) # Disable save if a new file is selected
            self.transcript_display.clear()
            self.transcript_segments = []
            logger.info(f"File selected: {filepath}")
        else:
            logger.info("File selection cancelled.")

    def start_transcription(self):
        """Initiates the transcription process in a background thread."""
        if not self.input_filepath:
            QMessageBox.warning(self, "No File", "Please select a file first.")
            return

        if self.transcription_thread and self.transcription_thread.isRunning():
            QMessageBox.warning(self, "In Progress", "Transcription is already running.")
            return

        # Disable UI elements during processing
        self.transcribe_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.transcript_display.setPlaceholderText("Transcription in progress...")
        self.transcript_display.clear()
        self.status_bar.showMessage("Starting transcription...")

        selected_model = self.model_combo.currentText()

        # Create worker and thread
        self.transcription_thread = QThread()
        self.worker = TranscriptionWorker(self.input_filepath, selected_model)
        self.worker.moveToThread(self.transcription_thread)

        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_results)
        self.worker.error.connect(self.handle_error)
        self.worker.model_loaded.connect(self.worker.run_transcription) # Start transcription after model loads

        # Connect thread management signals
        self.transcription_thread.started.connect(self.worker.load_model) # Start loading model when thread starts
        self.transcription_thread.finished.connect(self.on_thread_finished) # Cleanup

        # Start the thread
        self.transcription_thread.start()
        logger.info("Transcription thread started.")


    def update_progress(self, value: int, message: str):
        """Updates the progress bar and status bar."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def handle_results(self, segments: List[TranscriptSegment]):
        """Processes the transcript segments received from the worker."""
        self.transcript_segments = segments
        formatted_transcript = "\n".join(seg.to_simple_format() for seg in segments)
        self.transcript_display.setText(formatted_transcript)
        self.save_button.setEnabled(True)
        self.status_bar.showMessage("Transcription Complete!", 5000) # Message disappears after 5s
        logger.info("Transcription finished successfully.")
        # Thread will finish itself, triggering on_thread_finished

    def handle_error(self, error_message: str):
        """Displays an error message."""
        QMessageBox.critical(self, "Transcription Error", error_message)
        self.status_bar.showMessage(f"Error: {error_message}", 5000)
        logger.error(f"Transcription failed: {error_message}")
        # Ensure UI is reset even on error
        self.on_thread_finished()


    def on_thread_finished(self):
         """Cleans up after the thread finishes (normally or via error)."""
         logger.info("Transcription thread finished.")
         self.progress_bar.setVisible(False)
         self.transcribe_button.setEnabled(True)
         self.select_button.setEnabled(True)
         self.model_combo.setEnabled(True)
         # Enable save button only if results were successful
         self.save_button.setEnabled(bool(self.transcript_segments))

         # Clean up thread and worker objects
         if self.transcription_thread:
             self.transcription_thread.quit()
             self.transcription_thread.wait()
         self.transcription_thread = None
         self.worker = None


    def save_transcript(self):
        """Saves the transcript segments to an SRT file."""
        if not self.transcript_segments:
            QMessageBox.warning(self, "No Transcript", "There is no transcript data to save.")
            return

        if not self.input_filepath: # Should not happen if save is enabled, but good check
            return

        # Suggest a default filename based on the input file
        input_path_obj = Path(self.input_filepath)
        default_save_name = f"{input_path_obj.stem}_transcript.srt"
        default_dir = str(input_path_obj.parent)

        save_filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript As", os.path.join(default_dir, default_save_name), "SRT Files (*.srt);;All Files (*)"
        )

        if save_filepath:
            try:
                with open(save_filepath, 'w', encoding='utf-8') as f:
                    for segment in self.transcript_segments:
                        f.write(segment.to_srt_format() + "\n") # Extra newline between entries
                self.status_bar.showMessage(f"Transcript saved to {os.path.basename(save_filepath)}", 5000)
                logger.info(f"Transcript saved successfully to: {save_filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")
                logger.error(f"Failed to save transcript: {e}", exc_info=True)
        else:
            logger.info("Save operation cancelled.")

    def closeEvent(self, event):
        """Ensure thread is stopped if window is closed."""
        if self.transcription_thread and self.transcription_thread.isRunning():
            logger.warning("Attempting to close window while transcription is running. Stopping thread.")
            # You might want to ask the user for confirmation here
            self.transcription_thread.quit() # Ask thread to stop
            self.transcription_thread.wait(2000) # Wait a bit for clean exit
            if self.transcription_thread.isRunning(): # Force terminate if still running
                 logger.warning("Thread did not quit gracefully, terminating.")
                 self.transcription_thread.terminate()
                 self.transcription_thread.wait()


        event.accept() # Accept the close event

if getattr(sys, 'frozen', False):
    base_path = Path(sys.executable).parent
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))
    site_packages_path = base_path / 'Lib' / 'site-packages'
    if site_packages_path.is_dir() and str(site_packages_path) not in sys.path:
         sys.path.insert(0, str(site_packages_path))
         
# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = TranscriptionApp()
    main_window.show()
    sys.exit(app.exec())