from pathlib import Path
p=Path('Report/GeometricFidelity_2026-11-02/verify_statistics.py');s=p.read_text().replace("for typ in ['same_writer_wrong_character','other_writer_same_character']:","for typ in (['same_writer_wrong_character','other_writer_same_character'] if kind=='character' else []):");p.write_text(s)
