from dgplib.core import SignalBurnManager

INPUT_FOLDER = "cha1/2026-07-10T12-00-00/"  # Upewnij się, że to faktyczna ścieżka
OUTPUT_FOLDER = "cha1/bins"
DATASET = "rf_data"     # To musi być wewnętrzna nazwa datasetu w pliku H5!
N_POINTS = 2048

manager = SignalBurnManager()
manager.run_batch(INPUT_FOLDER, OUTPUT_FOLDER, DATASET, N_POINTS)
print("Done")