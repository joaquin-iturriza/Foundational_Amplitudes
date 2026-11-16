---
name: paper-writing-style
description: >-
  How to write paper text for Foundational_Amplitudes: forward argumentation, sharp physics-first arguments,
  sentence and paragraph patterns, vocabulary, and the equation, figure, caption and reference conventions. Use it
  whenever writing, rewriting, editing or reviewing LaTeX paper text (sections, results paragraphs, captions,
  abstracts, conclusions, outlooks) for the amplitude papers, and for the prose of docs/results.tex.
---

# Paper writing style

The group's writing rules, merged with our own. The voice to match is that of our papers, arXiv:2601.13308
(scaling laws for amplitude surrogates), 2405.14806 and 2411.00446: read them before writing, and plot as they do
(figure-style skill). The text is for a reader who knows only the paper. Never put people's names into text about
these rules. CLAUDE.md "Results and reporting" holds the process rules (reviewer before Joaquin, which sections may
change); this skill holds how the text reads.

## The core rules

1. **Forward argumentation.** "We want Y, so we do X", never "we do X, because that way we can Y". Check the
   direction of every sentence that explains why something is done, and flip it if it runs backwards. Motivation
   comes before method.
2. **Sharp arguments only.** Every sentence earns its place. Cut hedging, throat-clearing, the obvious, and anything
   the equation or figure already says.
3. **Formulas work without the text.** If extra notation seems needed, simplify the argument, not the words. No
   proofs.
4. **A symbol is defined inline at its first use, and only if it is used afterwards.** No notation tables.
5. **Physics first, method second.** The readers are physicists and master's students.
6. **Fewer words.** Each piece of information once, tersely.
7. **Active voice, present tense:** "we train", "we find". Passive only for objective, universal statements; past
   tense only in the outlook or conclusions.
8. **No internal names and no history.** No run, sweep, arm or recipe names, no "Version 1", no compile or trial
   counts, no account of how a result was obtained or what was tried first. A thing is named by what it is (a
   process by its reaction, a pretraining by what it trained on). The same rule for figures is enforced by
   `.claude/hooks/send_check.py`, with accepted terms in `docs/glossary.txt`.
9. **A result is shown in a plot or not at all.** A value read off a plotted figure counts as shown and is stated
   where it makes a point: a gain read from two plotted curves (fine-tune against scratch), a count of plotted
   points, a position on an axis, with the figure cited. What no figure shows (a number from a run, an aggregate
   over several figures, a spread never drawn) is not stated: plot it, or drop it and state the problem. A result
   that can only be given qualitatively is probably wrong or not important enough: rethink why it cannot be
   quantified. `docs/results.tex` is a record: there every claim carries its number (notes-editor).
10. **No hedging in the final text.** A claim that cannot be defended is not made. Hedging stays only where it is by
   design, in work in progress, which does not go into the final paper.

## Flow, not templates

The patterns below are tendencies, not slots. If every sentence runs "For X, verdict. The reason is ...", the text
is monotonic. Connect paragraphs as an argument:
- **A question:** "the question is if ...", "Once we are convinced that ..., the question is ...".
- **The figure as the answer:** "In the left panel of Fig.~\ref{...} we show ...", "In the right panel ... and
  confirm that ...".
- **Consequences:** "This means that ...".
- **Comparisons:** "As for ...", "While ...", "However, compared to ...", "Again, ...", "Consequently, ...".

Prose paragraphs, not bullet lists, for results.

## Sentence patterns

- **"For [context], [verdict]."** The context (process, pretraining, data size, phase-space region) first, the
  verdict after, without "we observe that". Example: "For $e^+e^-\to b\bar b$ at one loop, the pretrained network
  reaches the scratch network's loss with a tenth of the data."
- **"Again, ..."** opens a sentence when a pattern repeats, instead of explaining it again.
- **"... provides us with ..."** carries the payoff sentence of a case block.
- **Limitations are appended, never foregrounded:** a subordinate clause after a positive claim ("..., until the
  loss is dominated by the numerical precision of the amplitudes."). Never "Although X is limited by ..., it ...".
- **"The reason is that ..."** gives a real cause its own sentence, at most once per section and never in two
  neighbouring paragraphs. Usually the forward order (setup first, then result) needs no connective.
- **A colon for elaboration** instead of "namely", "specifically", "in particular".
- **"familiar"** tells the reader a pattern is one they know, without explaining it.
- **Rhythm:** a short declarative sentence (under ~15 words) carries the claim, a longer one carries the mechanism,
  chaining at most two clauses, then reset. A reader who skims only the short sentences still gets the argument.
- **"Finally, ..."** only for the closing sentence of a results paragraph, carrying the full-range claim.

## Paragraph and section structure

- **Sections** open with a declarative problem statement or physics goal, never with a definition.
- **Results paragraphs:** what the figure shows or what is tested; the methods in order of increasing
  sophistication; the "Finally, ..." sentence.
- **"We find that ..."** introduces empirical findings. No conclusions announced in the methods part.
- **Conclusions:** the physics problem, with no "in this work we ..."; one sentence on the approach ("In this paper,
  we have targeted this problem with ..."); a two-sentence result per case; an outlook that follows from the
  results, limitations appended to positive claims.

## Vocabulary

| Avoid | Write |
|---|---|
| "neural-network" (as a noun) | "neural network" |
| "performs well", "shows good results", "is able to" | maintains, recovers, achieves, provides us with, follows (results); fails, inherits, collapses, drops (failures); find, observe, confirm, show (observations) |
| "In this work we consider/explore/apply X" | cut, and state the approach directly |
| vague hedges ("shows good performance, but typically reaches limitations when ...") | cut, and give the per-case result |
| "crucial(ly)", "powerful", "seamless", "notably", "importantly", "it is worth noting", "remarkably", "striking" | cut |
| "not just X, but Y"; rule-of-three padding; "studies show" with nothing behind it | a plain declarative; the one item that matters; a citation |
| an em-dash (`---`), above all in a header | a comma, colon or full stop |
| "surrogate model" | "surrogate network" |
| "scaled" (loss) | "heteroscedastic" |

**Also cut:** sentences that restate an equation; what the figure under discussion already shows; negative results
implied by the main result.

## Equations, references, figures

- **Around an equation:** before it, 0-2 sentences with the goal; after it, the symbol definitions inline, one
  sentence with the implication, then stop.
- **References:** equations `Eq.~\eqref{eq:..}` (`Eqs.~\eqref{eq:a} and \eqref{eq:b}`), never `\cref`; figures
  `Fig.~\ref{fig:..}`, sections `Sec.~\ref{sec:..}`, tables `Tab.~\ref{tab:..}`, also at the start of a sentence.
- **Comparisons with other work** are made, and framed without an adversarial tone: what we do and what it gives
  beside the reference result, never "Unlike Ref. X, we ..." set up as a contest.
- **No `\emph{}`** anywhere: the term in plain text.
- **`\text{}`, not `\mathrm{}`, for words inside math:** `A_\text{NN}`, `\tau_\text{lo}`.
- **No subfigures.** A plain grid of `\includegraphics` (`\widerow` in the draft), each panel's position described
  in the one caption. No panel titles (figure-style skill).
- **Every caption stands alone:** the process, the setting and every legend symbol defined, the methods ordered as
  in the text or its reverse. Never "As Fig.~\ref{...}, for ...": write the full caption, even if it repeats another. What the figure shows goes in the caption, never inside the plot (figure-style).

## Before you hand back text

Check every sentence, also of a one-line fix: it argues forward; it chains at most two clauses; it says something
new; it has no hedging, meta-commentary, adversarial comparison or internal name; every result in it is plotted; every symbol in it is defined and used; it uses no
`\emph`, no `\mathrm` for words and no `\cref` for equations; every caption stands alone.
