import streamlit as st


st.write("You are an assistant that reads company price-increase announcements and classifies the justification for the price increase using exactly one of the following nine labels:")

st.write("1. Cost")
st.write("2. Quality")
st.write("3. Market")
st.write(("4. No-justification") 
st.write("5. Other")  
st.write("6. Cost, quality")
st.write("7. Cost, market") 
st.write("8. Market, quality")
st.write("9. Cost, market, quality")

st.write("Below are the classification rules. Use them strictly and consistently.")

st.write("**COST**\n  
Assign this label when the price increase is attributed to **rising expenses** the firm incurs to operate. This includes any of the following:  
- labor, materials, fuel, packaging, freight, logistics, insurance, maintenance  
- general operations, compliance costs, regulatory fees, or energy  
- macroeconomic pressures such as inflation, tariffs, or taxation  
The firm may describe cost directly (`higher labor costs`) or abstractly (`rising costs,` `cost pressures,` or `inflation`).  
**Do not assign cost** if the only justification is about delivering, preserving, or enhancing customer value.
**Do not assign cost** when the firm frames spending as part of value creation, investment, or strategic improvement unless operational cost drivers (e.g., labor, materials, inflation) are also explicitly mentioned.
**QUALITY**  
Assign this label only if the firm claims that the price increase is due to **improved customer-facing value**. This includes:  
- added features, better service, more convenience, upgraded performance, expanded access, or premium offerings  
- statements that frame the increase as necessary to **maintain a high standard of quality**, where the tone is clearly about **preserving elevated value for the customer**  
- references to content improvements, product upgrades, or infrastructure expansion, when framed as increasing or sustaining customer value
- investments in experience or innovation, if presented as a benefit to the customer  
**Do not assign quality** if the explanation refers to cost of operations, sustainability, or business survival without framing them as customer-facing improvements.
If the firm mentions increased expenses to deliver, maintain, or improve customer value, classify as quality only — not cost + quality.
These are framed as value-enhancing efforts, not burdensome operational costs, and should not trigger a cost label unless cost drivers (like labor, materials, or inflation) are mentioned separately.

**MARKET**  
Assign this label only if the justification is due to **external demand or competitive dynamics**. This includes:  
- increased usage, consumer demand, seasonal surges, or growth in user base  
- limited availability of the firm’s own offerings **due to constrained supply by rivals, or high market uptake**  
- explicit or implied alignment with competitor prices, or reference to `market rates`
**Do not assign market** if the statement only refers to upstream supply issues (e.g., chip shortages), general volatility, or unfavorable conditions.
If the justification mentions shortages of inputs (e.g., chips, raw materials), this is a cost driver — not market.

**NO-JUSTIFICATION**  
Assign this label if the letter **does not provide any reason** for the price change.  
- Merely describing what will change or when the price will change is not a justification.  
- Statements like `we are updating our rates` or `we are committed to transparency` alone do not count.

**OTHER**  
Assign this label if the justification does not fit cost, quality, or market. This includes:
- Vague references to `business sustainability,` `uncertain economic environment,` `strategic realignment,` `long-term goals,` `corporate restructuring,` or `charitable giving,` without tying them to cost, quality, or market drivers.
- Mergers, rebranding, or shifts in business model not presented as customer improvements.

**COMBINATION LABELS**  
Assign a combination label only when the letter **clearly presents more than one justification** type.  
- If a letter includes both a rising operating cost (e.g., inflation) and added customer value (e.g., new features), assign `Cost, quality`.  
- Do not assign `Cost, quality` simply because the letter says `we’ve invested heavily to provide a better experience.` Unless operational cost increases are also explicitly mentioned, this is quality only.
- Do not assign combinations based on vague or emotional language — each component must be independently and explicitly present.

Your task: Read the following letter and return **only the most appropriate label** from the list above. Do not include any explanation, reasoning, or formatting. Return the label exactly as written.
Respond with exactly one label (from the nine above) and nothing else.")
