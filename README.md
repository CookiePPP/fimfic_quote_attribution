# Fimfiction Quote Atrribution using DistilBERT and Masked-Language Modelling Objective.

This repo is currently abandoned. The developer is busy with other stuff.

The best model achieves 88% accuracy and works really well for making fully voiced AI Audiobooks, however it's got a lot of small bugs I need to fix before it's good for general usage.

---

Because the model predicts the easiest tokens first and uses it's own prediction as labels, the model is very consistent even when it's wrong.

For example, in one story there is an unnamed Librarian character, and the model automatically gave that character the voice of Twilight Sparkle and plays the entire scene with her acting in place of the actual Librarian.

----

TODO

- [ ] Remove samples with "the {X} {said}"
  Most examples with 'the' prefix are not names and don't have assigned voices.

- [ ] Use 'she'/'he' prefix to constrain model outputs during inference and allow user to specify the genders of OC's and characters.
