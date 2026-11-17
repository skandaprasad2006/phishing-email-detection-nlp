# Error Analysis

Generated: 2026-11-17T20:20:26  Run tag: `smoke`

Misclassifications on the holdout test set, described by measured properties
only. Each statement below is a count or a rate over the actual errors; no
cause is asserted that the data does not show.

| Model | Error type | Count | Mean chars | Median chars | Mean words | Share with a URL | Share under 200 chars |
|---|---|---|---|---|---|---|---|
| Random Forest | False positive (legitimate marked phishing) | 31 | 142 | 142 | 22 | 0.0% | 93.5% |
| Random Forest | False negative (phishing missed) | 37 | 138 | 147 | 22 | 0.0% | 100.0% |
| Logistic Regression | False positive (legitimate marked phishing) | 23 | 146 | 142 | 22 | 0.0% | 95.7% |
| Logistic Regression | False negative (phishing missed) | 30 | 136 | 147 | 22 | 0.0% | 100.0% |

Reference: correct-classification length means, per model.

- Random Forest: mean 138 characters across the test set.
- Logistic Regression: mean 138 characters across the test set.

## Tokens over-represented in each error class

### Random Forest

**Most frequent tokens among false positives:** `details` (25), `thanks` (24), `schedule` (23), `today` (21), `team` (21), `meeting` (21), `note` (21), `urltoken` (20), `review` (20), `report` (17), `regards` (15), `updated` (14), `information` (12), `week` (10), `claim` (7)

**Most frequent tokens among false negatives:** `week` (24), `updated` (24), `today` (23), `thanks` (23), `regards` (23), `information` (22), `review` (21), `meeting` (19), `details` (18), `schedule` (17), `report` (15), `team` (14), `note` (13), `urltoken` (9), `let` (9)

**Most frequent tokens among correctly detected phishing:** `urltoken` (51), `details` (40), `report` (35), `meeting` (32), `today` (30), `information` (29), `week` (28), `team` (26), `updated` (26), `schedule` (26), `note` (26), `review` (24), `regards` (22), `thanks` (22), `verify` (14)

- False positive phishing probability: mean 0.6725, range 0.52-0.96.
- False negative phishing probability: mean 0.2787, range 0.0-0.48.

### Logistic Regression

**Most frequent tokens among false positives:** `details` (24), `urltoken` (22), `schedule` (18), `note` (18), `today` (16), `team` (15), `report` (13), `updated` (13), `review` (13), `meeting` (13), `regards` (12), `thanks` (12), `information` (9), `week` (8), `claim` (7)

**Most frequent tokens among false negatives:** `week` (23), `review` (20), `thanks` (18), `today` (18), `regards` (18), `updated` (18), `information` (17), `schedule` (16), `details` (15), `note` (13), `meeting` (13), `team` (11), `let` (10), `know` (10), `report` (10)

**Most frequent tokens among correctly detected phishing:** `urltoken` (60), `details` (43), `report` (40), `meeting` (38), `today` (35), `information` (34), `updated` (32), `week` (29), `team` (29), `regards` (27), `schedule` (27), `thanks` (27), `note` (26), `review` (25), `claim` (18)

- False positive phishing probability: mean 0.6601, range 0.5019-0.7776.
- False negative phishing probability: mean 0.3154, range 0.1951-0.4979.

## Masked examples

Addresses and URLs are redacted and excerpts truncated, so this report can be
shared without exposing message content. Full masked rows are in
`results/error_samples_*.csv`.

### Random Forest

**False positives**

- `the deployment finished successfully and the dashboard shows normal latency figures please updated thanks regards thanks schedule team note review team report`
- `action required mailbox quota exceeded verify your password now through urltoken today review details review please`
- `lunch is at one in the cafeteria let me know if you want to join the group`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials regards today thanks please`
- `the deployment finished successfully and the dashboard shows normal latency figures please details regards meeting note details regards please team`
- `action required mailbox quota exceeded verify your password now through urltoken today note meeting thanks updated review team updated note details thanks updat...`
- `hi team attached are the minutes from our planning meeting please review before friday regards schedule thanks meeting note schedule`
- `thanks for your message i will send the updated schedule after the review call tomorrow team updated team thanks`
- `notes from the lecture are uploaded to the course page see section three for derivatives information regards today today meeting information information please ...`
- `urgent final notice your payment failed click here urltoken confirm your billing details now updated thanks thanks meeting`

**False negatives**

- `reminder the quarterly report is due next week let me know if you need the template information details today team note today thanks regards regards review revi...`
- `please find the invoice for last month attached and let me know about the purchase order team information review note updated schedule schedule report`
- `please find the invoice for last month attached and let me know about the purchase order team week regards meeting week schedule updated thanks team updated tod...`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials information`
- `notes from the lecture are uploaded to the course page see section three for derivatives schedule today information information regards`
- `the library renewed our journal subscription so the older issues are accessible again today schedule please today report meeting report regards today`
- `the deployment finished successfully and the dashboard shows normal latency figures review note meeting week please week information regards updated review plea...`
- `notes from the lecture are uploaded to the course page see section three for derivatives details team week`
- `the library renewed our journal subscription so the older issues are accessible again thanks regards details thanks information note today please updated`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials schedule meeting meeting team team regards updated report meeting week ...`

### Logistic Regression

**False positives**

- `action required mailbox quota exceeded verify your password now through urltoken today review details review please`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials regards today thanks please`
- `action required mailbox quota exceeded verify your password now through urltoken today note meeting thanks updated review team updated note details thanks updat...`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials information`
- `urgent final notice your payment failed click here urltoken confirm your billing details now updated thanks thanks meeting`
- `security alert unusual sign in detected confirm your identity at urltoken or lose access week report note note please`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials schedule meeting updated please note week meeting team team schedule te...`
- `you have won a prize claim your reward within hours visit urltoken and enter credentials thanks note report schedule details information note`
- `your bank account is locked please revalidate your card details at urltoken urgently details week details please report`
- `your bank account is locked please revalidate your card details at urltoken urgently review team report regards`

**False negatives**

- `reminder the quarterly report is due next week let me know if you need the template information details today team note today thanks regards regards review revi...`
- `please find the invoice for last month attached and let me know about the purchase order team information review note updated schedule schedule report`
- `please find the invoice for last month attached and let me know about the purchase order team week regards meeting week schedule updated thanks team updated tod...`
- `notes from the lecture are uploaded to the course page see section three for derivatives schedule today information information regards`
- `the library renewed our journal subscription so the older issues are accessible again today schedule please today report meeting report regards today`
- `the deployment finished successfully and the dashboard shows normal latency figures review note meeting week please week information regards updated review plea...`
- `notes from the lecture are uploaded to the course page see section three for derivatives details team week`
- `the library renewed our journal subscription so the older issues are accessible again thanks regards details thanks information note today please updated`
- `notes from the lecture are uploaded to the course page see section three for derivatives meeting updated please today week updated information note`
- `lunch is at one in the cafeteria let me know if you want to join the group meeting details please today regards details report updated updated schedule`

## What this analysis does and does not show

It shows the measured profile of the mistakes: how many, how long, how often
they contain URLs, and which tokens dominate. It does not establish why any
individual email was misclassified; that would require labelling the error
causes manually, which has not been done.
