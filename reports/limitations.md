# Known Limitations & Findings

Captured from feature importance and error analysis (Day 8), for reuse in the
Week 3 comparative report.

## Data Quality

- **A small number of rows have placeholder/empty body content** (literal
  "empty" text or similarly low-signal placeholders) rather than a real
  email body. These appear among the false positives — with no real content
  to key off, the model has nothing reliable to classify and lands just
  over the decision threshold. This is a source-dataset characteristic, not
  a pipeline bug. Documented here rather than filtered out, since it affects
  a small fraction of the 78 false positives and re-running the full
  pipeline to chase it isn't a good time/benefit tradeoff at this stage.

- **The "safe" class is dominated by Enron corporate/academic vocabulary**
  (`enron`, `vince`, `university`, `linguistics`, `edu`) rather than generic
  everyday email. This is a known characteristic of the classic Enron +
  phishing combo dataset. It means the model likely learned "is this
  Enron-flavored corporate/academic English" as a strong proxy for "safe,"
  which may not generalize as well to other legitimate email sources (e.g.
  personal Gmail traffic). Worth stating explicitly as a generalization
  caveat rather than implying the model detects "legitimate email" in
  general.

## Feature Importance Findings

- **Logistic Regression coefficients validate the project brief's core
  hypothesis**: top phishing-indicative words are `click`, `remove`, `free`,
  `http`, `money`, `site`, `removed`, `info` — classic urgency/sales/
  call-to-action language.
- **Random Forest confirms the engineered metadata features earn their
  place**: `exclamation_count` and `caps_ratio` rank among the single most
  important features, on par with top TF-IDF words — validates that
  metadata feature engineering (Day 2) added real signal, not just noise.

## Error Analysis Findings (Neural Network)

False negatives (phishing that slipped through) cluster into distinct
patterns, not random noise:

1. **Possible label noise at the spam/phishing boundary** — a corporate-
   newsletter-style false negative reads more like general spam than
   credential-theft phishing; the line between the two is genuinely blurry
   even for human labelers.
2. **Link-farm / MLM-style spam** — spammy but lacking urgency language,
   a different phishing "flavor" than the click-bait style the model
   learned best from.
3. **Content obfuscation** — one false negative was padded with unrelated,
   plausible-sounding text (historical quotes, unrelated trivia) with no
   suspicious vocabulary present. This looks like a deliberate evasion
   technique: stuffing irrelevant text to dodge keyword-based detection.
   This is a structural blind spot of TF-IDF-based models — a real
   limitation worth naming rather than treating as a random miss.
   Word-embedding or transformer-based models that capture context rather
   than just term frequency would be a natural next step to address this
   specific failure mode.

## Suggested one-liner for the report

> "The model's false negatives cluster around content-obfuscation-style
> phishing, where attackers pad messages with unrelated text to avoid
> triggering keyword-based detection — suggesting TF-IDF-based classifiers
> have a structural blind spot that context-aware models (e.g.
> transformers) could potentially address."