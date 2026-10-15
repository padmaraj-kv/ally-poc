from ally import prompts
from ally.schemas import Segment, SessionNote


def note_items(note: SessionNote) -> list:
    return [*note.themes, *note.concerns, *note.progress, *note.follow_ups, *note.tone]


async def generate_note(llm, segments: list[Segment], language: str) -> tuple[SessionNote, list[str]]:
    known = {s.id for s in segments}

    def check(note: SessionNote) -> list[str]:
        bad = sorted({i for item in note_items(note) for i in item.segment_ids if i not in known})
        return (
            [f"unknown segment ids cited: {', '.join(bad)}; cite only ids shown in the transcript"]
            if bad
            else []
        )

    note = await llm.structured(
        prompts.NOTES_SYSTEM,
        prompts.notes_user(language, segments),
        SessionNote,
        effort="medium",
        check=check,
    )
    removed = 0
    for item in note_items(note):
        kept = [i for i in item.segment_ids if i in known]
        removed += len(item.segment_ids) - len(kept)
        item.segment_ids = kept
    return note, [f"{removed} unknown citation(s) removed from the note"] if removed else []
