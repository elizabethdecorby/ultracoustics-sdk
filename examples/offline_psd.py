"""Offline PSD (power spectral density) processing example for Ultracoustics Python SDK."""

from ultracoustics import compute_psd, load_binary


def main() -> None:
    samples = load_binary("capture.bin")
    freq_hz, psd_db = compute_psd(samples, fft_size=8192, num_averages=20)

    print(f"Loaded {len(samples)} samples")
    print(f"PSD bins: {len(psd_db)}")
    print(f"Frequency range: {freq_hz[0]:.1f} Hz to {freq_hz[-1]:.1f} Hz")


if __name__ == "__main__":
    main()
