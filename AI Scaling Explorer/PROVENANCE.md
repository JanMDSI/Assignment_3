# Provenance matrix

Each team member adds their own rows. A row says what you contributed, where it came from, one check you ran that could have failed, and what happened. A check like "looked fine" does not count. A good check has a number or a pass or fail outcome that someone else could repeat.

How to use it: replace the bracketed parts, delete the examples that do not apply to you, and add rows for anything else you made or checked. Every team member needs at least one row.

| Member | Item | Where it came from | Check that could have failed | Result and date |
|---|---|---|---|---|
| [Member 1 name] | data/raw/epoch_ai_models.csv | Downloaded from epoch.ai/data/ai-models | Compare the CSV's row count with the row count Epoch shows for the same download. A truncated download would give a lower count. | [pass or fail, count, date] |
| [Member 1 name] | Cleaning steps in data/processed/cleaning_steps.csv | src/data.py | Open the raw file in a spreadsheet, filter to rows with both Parameters and Training dataset size filled, and compare with the "reported dataset size" step. The counts must match. | [pass or fail, date] |
| [Member 2 name] | Five rows of data/processed/models.csv | Epoch AI, traced to the paper | Pick five models at random, open each source link, and compare parameters and dataset size with the paper. Any mismatch is a fail. | [models checked, pass or fail, date] |
| [Member 2 name] | Growth split for one slice in growth_by_slice.csv | src/analysis.py | In a spreadsheet, take log10 of parameters and of dataset size for the Language, 2010 onward slice and use SLOPE against release year. The two shares must match the file to two decimal places. | [pass or fail, values, date] |
| [Member 3 name] | Charts and colours | src/views.py | Screenshot the scatter plot and run it through Coblis or Color Oracle for red green and blue yellow colour blindness. Fail if two categories cannot be told apart by colour or shape. | [tool used, pass or fail, date] |
| [Member 3 name] | Load time of the deployed app | Streamlit Community Cloud | Open the deployed URL in a private window with developer tools open. Fail if the page is not usable within five seconds. | [seconds, date] |
| [Member 4 name] | Layout at 1366 by 768 | Browser responsive mode | Set the viewport to 1366 by 768 and scroll the whole page. Fail if a horizontal scrollbar appears or any chart is cut off. | [pass or fail, date] |
| [Member 4 name] | The three user stories | README.md | Someone who did not build the app follows each walkthrough in README.md without help. Fail if they get stuck or the screen does not match the text. | [who, pass or fail per story, date] |
| [Member 1 name] | Automated tests | tests/ | Run pytest on a clean checkout of the submitted commit. Fail if any test fails. | [tests passed out of total, date] |
| [Member 1 name] | AI generated text and code | AI_DECLARATION.md | For every on screen sentence with a number, find the number in data/processed/ or work it out. Fail if any figure was typed by hand. | [sentences checked, pass or fail, date] |
