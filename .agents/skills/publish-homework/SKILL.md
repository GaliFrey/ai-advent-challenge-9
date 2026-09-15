---
name: publish-homework
description: Publish an AI Advent Challenge result when the user says «опубликуй видео с результатом», «опубликуй день», «опубликуй ДЗ» or asks to upload a day's video or solution. Handles verified Yandex Disk upload through rclone, homework checks, README conclusions, Git commit/push, and final links according to the requested mode.
---

# Publish Homework

Publish the active AI Advent Challenge day without confusing video upload with Git publication. Treat the user's matching publication request as authorization for the external writes explicitly included in that mode. Do not ask for the same authorization again, though host-level approval may still be required by the execution environment.

## Resolve the target day

Determine one `day-XX` from, in order:

1. an explicit day in the current request;
2. the day being implemented or discussed in the current Codex task;
3. an unambiguous current working directory inside a `day-XX` folder.

Do not select the numerically latest directory merely because it exists. If the conversation and working directory do not identify exactly one active day, ask the user which day to publish before making external changes.

Resolve the repository root with Git. Before publishing, inspect rather than assume:

- current branch and upstream;
- `origin` fetch and push URLs;
- working-tree changes;
- configured rclone remotes.

The established project convention is:

- local video: `day-XX/resources/day-XX-result.mp4`;
- rclone remote: `yadisk:`;
- remote object: `yadisk:AI Advent Challenge 9/day-XX/day-XX-result.mp4`;
- Git repository: `GaliFrey/ai-advent-challenge-9`;
- normal publication branch: `main`.

Use the discovered values when they differ. Do not silently publish to an unexpected repository, branch, remote, day, or filename; stop and report the mismatch.

## Select the requested mode

### «Опубликуй видео с результатом»

Perform only the video workflow. Do not edit README, commit, or push unless the user also asked to publish the day or homework.

Return the verified public video link. Return a README link too only if that README is already published and the user explicitly requested both links.

### «Опубликуй день»

Perform the solution and Git workflow. A verified video on Yandex Disk is a mandatory precondition. If it is absent remotely but the local video exists, upload and verify it as part of this mode so the README can contain a valid link.

Return exactly two primary links: the video and the published README.

### «Опубликуй ДЗ»

Perform the complete workflow: video, solution, README, commit, push, and post-publication verification.

Return exactly two primary links: the video and the published README.

## Video workflow

1. Locate the expected video inside the target day. Search that day recursively only when the conventional path is absent.
2. Require exactly one intended result video. If none exists, stop. If multiple plausible files exist, identify them and ask which one is the result instead of guessing.
3. Verify that the local file is non-empty. Use `ffprobe` to record container, codecs, dimensions, duration, and stream count. Do not claim visual inspection unless it was actually performed and authorized by the applicable project instructions.
4. Compute local size, MD5, and SHA-256 without printing unrelated file contents.
5. Inspect the remote object before upload. Upload with `rclone copyto` to the resolved full object path; do not use a broad directory sync.
6. Read remote metadata after upload and require the remote object to exist. Compare size. Compare each checksum the provider exposes; state explicitly when a checksum is unavailable rather than treating it as a match.
7. Obtain a public link with rclone. Verify it without authentication when possible. For a Yandex Disk public link, use its public metadata endpoint when available to confirm the expected filename, media type, size, and antivirus status.
8. The video workflow is incomplete until both the remote object and usable public link are verified. A successful upload command alone is not proof of publication.

Never print rclone configuration, tokens, API keys, cookies, or signed download URLs. The stable public page URL is safe to return.

## Solution and Git workflow

1. Read the target day's assignment, README, applicable `AGENTS.md`, code, saved result artifacts, and relevant latest entries in `history.md`.
2. Inspect the working tree before edits. Preserve unrelated user changes and never stage them accidentally.
3. Review the implementation against the assignment. Run the project's documented checks with its required tooling. For Python days, use `uv` and verify `uv.lock`. Do not make paid or real LLM calls merely to publish; use already recorded results unless the user requested a new run.
4. If a required check fails, diagnose it. Fix only issues within the publication request and re-run the checks affected by the fix. Do not publish a knowingly failing solution.
5. Derive README results and conclusions from recorded evidence: saved responses, logs, test output, and the verified video. Do not invent metrics, observations, model behavior, or visual checks.
6. Update the day's README so it accurately contains:
   - what was implemented;
   - how to run it;
   - the factual experiment or demonstration result;
   - concise conclusions tied to those results;
   - the verified public video link;
   - relevant checks and honest limitations.
7. Update `history.md` with one concise entry for this completed publication iteration. Update the root README only if the day is missing or its description is stale.
8. Check the candidate diff, whitespace, ignored files, and accidental secrets. Videos, `.env`, chats, local JSON data, caches, and unrelated files must not enter Git.
9. Stage only the intended files for the active day and directly related repository documentation. Review the staged diff and staged file list before committing.
10. Create one focused commit and push the current intended branch to its configured upstream. A request to publish the day or homework authorizes this commit and push, but not unrelated changes or force-pushes.
11. Verify that the remote branch contains the new commit. Build the README URL from the actual Git remote, branch, and `day-XX/README.md`; verify that the published page is reachable when network tools permit.

Do not rewrite history, force-push, delete remote files, or change repository visibility as part of this workflow.

## Completion and stopping conditions

Publication is complete only when every requirement for the selected mode is verified.

For day or homework publication, require all of these:

- the correct local video exists;
- the corresponding object exists on Yandex Disk;
- local and remote size match;
- the public video link works;
- required solution checks pass;
- the README contains factual results, conclusions, and the verified video link;
- the intended commit exists on the remote branch;
- the final response contains the video link and README link.

If any required condition fails, do not say that publication succeeded. Report the exact incomplete step, evidence already verified, and the action needed to continue. Do not retry unchanged failures repeatedly.

## Final response

Lead with the outcome and briefly name the checks performed. For a completed day or homework publication, finish with exactly these two labeled links:

- `Видео: <public Yandex Disk URL>`
- `README: <published Git URL>`

Do not substitute a local path, repository root, commit URL, rclone object path, or temporary download URL for either required link.
