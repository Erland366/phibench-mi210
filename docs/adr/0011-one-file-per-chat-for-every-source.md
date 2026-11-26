# Every record keeps its chats in one place, one file per conversation, for every source

Status: accepted (2026-11-26); refines ADR 0008 and 0010

Every trace record keeps its conversations in `conversation/chats/`, one file per conversation, listed in
`record.json` `conversation.chats`: one chat (`01-solver.jsonl`) for a run with one agent, such as an open model's
run or a person's session, and one chat per agent for a GEAK run. Before this, an open-model record had its chat in
`conversation/messages.jsonl`, while a GEAK record had its chats in `conversation/agents/` and a `messages.jsonl` that
ran all its agents together, which looked like one dialogue that never happened; so where the chat was depended on
the source (user, 2026-11-26: the format has to be unified). There is no longer a file that runs several chats
together; a GEAK run stays one record (ADR 0010).

## Considered options

- One record per chat, splitting a GEAK run into one record per agent: every record would hold exactly one chat,
  but a GEAK run's label and its reference come from the whole run (ADR 0010), so its agents would repeat one label.
