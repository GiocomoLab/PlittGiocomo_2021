import traceback
import morph_analyses as m


def write_session(mouse, metadata, session, training_session=False):
    nwb_maker = m.create_nwb.SessNWBConverter(
        mouse, metadata, session,
        training_session=training_session,
    )
    if nwb_maker.out_path.exists():
        print(f"Skipping {mouse} day {session['day']} (already exists)")
        return

    nwb_maker.build_file()
    nwb_maker.write_file()
    print(f"Wrote {nwb_maker.out_path}")


def run_all():
    for mouse, metadata in m.mouse_metadata.rare_sessions.items():
        for session in metadata['test_sessions']:
            try:
                write_session(mouse, metadata, session, training_session=False)
            except Exception:
                print(f"ERROR: {mouse} day {session['day']} (test)")
                traceback.print_exc()

        for session in metadata['training_sessions']:
            try:
                write_session(mouse, metadata, session, training_session=True)
            except Exception:
                print(f"ERROR: {mouse} day {session['day']} (training)")
                traceback.print_exc()


if __name__ == "__main__":
    run_all()
