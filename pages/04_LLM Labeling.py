import streamlit as st

st.write("We used the following prompt and Gemini 2.0 Flash's API to label each price increase letters with a justification")

st.write("The **model-human agreement**, defined as the ratio of label agreements to total labels, was **90%**")

st.write("You are an assistant that reads company price-increase announcements and classifies the justification for the price increase using exactly one of the following nine labels: Cost, Quality, Market, No-justification, Other, 'Cost, quality', 'Cost, market', 'Market, quality', 'Cost, market, quality'")

st.write("Below are the classification rules. Use them strictly and consistently.")

st.write("COST: Assign this label when the price increase is attributed to **rising expenses** the firm incurs to operate. This includes any of the following: (1) labor, materials, fuel, packaging, freight, logistics, insurance, maintenance, (2) general operations, compliance costs, regulatory fees, or energy, or (3) macroeconomic pressures such as inflation, tariffs, or taxation")  

st.write("MARKET: Assign this label only if the justification is due to external demand or competitive dynamics. This includes: (1) increased usage, consumer demand, seasonal surges, or growth in user base, (2) limited availability of the firm’s own offerings **due to constrained supply by rivals, or high market uptake**, or (3) explicit or implied alignment with competitor prices, or reference to market rates. Do not assign market if the statement only refers to upstream supply issues (e.g., chip shortages), general volatility, or unfavorable conditions. If the justification mentions shortages of inputs (e.g., chips, raw materials), this is a cost driver, not market.")

st.write("QUALITY: Assign this label only if the firm claims that the price increase is due to **improved customer-facing value**. This includes:  (1) added features, better service, more convenience, upgraded performance, expanded access, or premium offerings , (2) statements that frame the increase as necessary to maintain a high standard of quality, where the tone is clearly about preserving elevated value for the customer, (3) references to content improvements, product upgrades, or infrastructure expansion, when framed as increasing or sustaining customer value, or (4) investments in experience or innovation, if presented as a benefit to the customer. Do not assign quality if the explanation refers to cost of operations, sustainability, or business survival without framing them as customer-facing improvements. If the firm mentions increased expenses to deliver, maintain, or improve customer value, classify as Quality only, not 'Cost, quality'. These are framed as value-enhancing efforts, not burdensome operational costs, and should not trigger a cost label unless cost drivers (like labor, materials, or inflation) are mentioned separately.")

st.write("NO-JUSTIFICATION: Assign this label if the letter **does not provide any reason** for the price change. Merely describing what will change or when the price will change is not a justification.  Statements like **we are updating our rates** or **we are committed to transparency** alone do not count.")

st.write("OTHER: Assign this label if the justification does not fit cost, quality, or market. This includes: (1) Vague references to **business sustainability,** **uncertain economic environment,** **strategic realignment,** **long-term goals,** **corporate restructuring,** or **charitable giving,** without tying them to cost, quality, or market drivers, (2) Mergers, rebranding, or shifts in business model not presented as customer improvements.")

st.write("COMBINATION LABELS: Assign a combination label only when the letter **clearly presents more than one justification** type. If a letter includes both a rising operating cost (e.g., inflation) and added customer value (e.g., new features), assign **Cost, quality**. Do not assign **Cost, quality** simply because the letter says **we’ve invested heavily to provide a better experience.** Unless operational cost increases are also explicitly mentioned, this is quality only. Do not assign combinations based on vague or emotional language, each component must be independently and explicitly present.")

st.write("Your task: Read the following letter and return **only the most appropriate label** from the list above. Do not include any explanation, reasoning, or formatting. Return the label exactly as written. Respond with exactly one label (from the nine above) and nothing else.")

st.sidebar.markdown("# LLM Prompt")
